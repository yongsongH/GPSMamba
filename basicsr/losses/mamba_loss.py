import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from basicsr.archs.irsrmamba_arch import SS2D as SS2D
from .basic_loss import L1Loss
_reduction_modes = ['none', 'mean', 'sum']
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.models import vgg19

@LOSS_REGISTRY.register()
class IRMambaLoss(nn.Module):

    def __init__(self, loss_weight=1.0, freq_weight=0.5, phase_weight=0.3):
        super().__init__()
        self.loss_weight = loss_weight
        self.freq_weight = freq_weight
        self.phase_weight = phase_weight
        self.vgg = vgg19(pretrained=True).features[:35]
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.freq_gate = nn.Sequential(nn.Conv2d(512, 256, 1), nn.ReLU(), nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.register_buffer('sobel_x', torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred, target):
        (phase_loss, thermal_mask) = self.cal_phase_thermal_loss(pred, target)
        freq_loss = self.cal_freq_loss(pred, target, thermal_mask)
        return self.loss_weight * (self.freq_weight * freq_loss + self.phase_weight * phase_loss)

    def cal_phase_thermal_loss(self, pred, target):
        pred_phase = torch.angle(torch.fft.rfft2(pred))
        target_phase = torch.angle(torch.fft.rfft2(target))
        phase_loss = F.l1_loss(pred_phase, target_phase)
        with torch.no_grad():
            thermal_feat = self.vgg(target)
            thermal_mask = self.freq_gate(thermal_feat)
        return (phase_loss, thermal_mask)

    def cal_freq_loss(self, pred, target, thermal_mask):
        thermal_mask = F.interpolate(thermal_mask, size=pred.shape[2:], mode='bilinear', align_corners=False)
        pred_freq = torch.fft.rfft2(pred * thermal_mask)
        target_freq = torch.fft.rfft2(target * thermal_mask)
        complex_diff = pred_freq - target_freq
        return torch.abs(complex_diff).mean()

    def rgb_to_grayscale(self, x):
        if x.size(1) == 3:
            return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return x.squeeze(1)

@LOSS_REGISTRY.register()
class MambaLoss(nn.Module):

    def __init__(self, loss_weight=1.0, freq_weight=0.1, grad_weight=0.05, tau=0.5, scales=(1, 2, 4)):
        super().__init__()
        self.loss_weight = loss_weight
        self.freq_weight = freq_weight
        self.grad_weight = grad_weight
        self.tau = nn.Parameter(torch.tensor(tau))
        self.scales = scales
        self.register_buffer('sobel_x', torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred, target):
        freq_loss = self.cal_freq_loss(pred, target)
        grad_loss = self.cal_grad_loss(pred, target)
        return self.loss_weight * (self.freq_weight * freq_loss + self.grad_weight * grad_loss)

    def cal_freq_loss(self, pred, target):
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        with torch.no_grad():
            mag = torch.abs(pred_freq)
            energy = mag.pow(2)
            norm_energy = energy / (energy.sum(dim=(2, 3), keepdim=True) + 1e-06)
            freq_weights = torch.sigmoid(norm_energy * self.tau)
        complex_diff = pred_freq - target_freq
        loss_map = torch.abs(complex_diff) * freq_weights
        return loss_map.mean()

    def cal_grad_loss(self, pred, target):
        total_loss = 0.0
        pred_grad = self.cal_spatial_gradient(pred)
        target_grad = self.cal_spatial_gradient(target)
        for s in self.scales:
            if s > 1:
                down_sample = nn.AvgPool2d(kernel_size=s, stride=s)
                p_grad = down_sample(pred_grad)
                t_grad = down_sample(target_grad)
            else:
                (p_grad, t_grad) = (pred_grad, target_grad)
            total_loss += F.l1_loss(p_grad, t_grad)
        return total_loss / len(self.scales)

    def cal_spatial_gradient(self, x):
        if x.size(1) > 1:
            x_gray = rgb_to_grayscale(x)
        else:
            x_gray = x
        g_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        g_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        return torch.sqrt(g_x.pow(2) + g_y.pow(2) + 1e-06)

    def extra_repr(self):
        return f'tau={self.tau.item():.2f}, scales={self.scales}'

@LOSS_REGISTRY.register()
class FlexibleCompositeLoss(nn.Module):

    def __init__(self, losses):
        super(FlexibleCompositeLoss, self).__init__()
        self.loss_funcs = nn.ModuleList()
        self.loss_weights = []
        self.loss_names = []
        for loss_config in losses:
            config = loss_config.copy()
            loss_type = config.pop('type')
            loss_weight = config.pop('loss_weight', 1.0)
            loss_class = LOSS_REGISTRY.get(loss_type)
            if loss_class is not None:
                self.loss_funcs.append(loss_class(**config))
                self.loss_weights.append(loss_weight)
                self.loss_names.append(loss_type)
            else:
                raise ValueError(f"Loss type '{loss_type}' not found in LOSS_REGISTRY.")

    def forward(self, pred, target):
        total_loss = 0.0
        for (i, loss_func) in enumerate(self.loss_funcs):
            loss_val = loss_func(pred, target)
            weighted_loss = self.loss_weights[i] * loss_val
            total_loss += weighted_loss
        return total_loss
