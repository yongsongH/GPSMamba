# [GPSMamba](https://arxiv.org/pdf/2507.18998)
Official PyTorch implementation of the paper [GPSMamba: A Global Phase and Spectral Prompt-guided Mamba for Infrared Image Super-Resolution.](https://arxiv.org/pdf/2507.18998)


## Introduction

Infrared (IR) image super-resolution faces challenges from homogeneous background pixel distributions and sparse target regions, requiring models that effectively handle long-range dependencies and capture detailed local-global information. Recent advancements in Mamba-based (Selective Structured State Space Model) models, employing state space models, have shown significant potential in visual tasks, suggesting their applicability for IR enhancement. In this work, we introduce IRSRMamba: Infrared Image Super-Resolution via Mamba-based Wavelet Transform Feature Modulation Model, a novel Mamba-based model designed specifically for IR image super-resolution. This model enhances the restoration of context-sparse target details through its advanced dependency modeling capabilities. Additionally, a new wavelet transform feature modulation block improves multi-scale receptive field representation, capturing both global and local information efficiently. Comprehensive evaluations confirm that IRSRMamba outperforms existing models on multiple benchmarks. This research advances IR super-resolution and demonstrates the potential of Mamba-based models in IR image processing.

## Approach overview

![GPSMamba](experiments/pretrained_models/GPSMamba.png)


## Requirements
> - Python 3.9, PyTorch >= 1.11
> - BasicSR 1.4.2
> - Platforms: Ubuntu 18.04, cuda-11



## Installation
>  Clone the repo
```
git clone https://github.com/yongsongH/GPSMamba.git
```
> Install dependent packages
```
cd GPSMamba
```
```
pip install -r requirements.txt
```
> Install BasicSR
```
python setup.py develop
```
***You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation***

## Dataset prepare

Please check this [page](https://figshare.com/articles/dataset/IRSRMamba_Infrared_Image_Super-Resolution_via_Mamba-based_Wavelet_Transform_Feature_Modulation_Model/25835938).

## Model

Pre-trained models can be downloaded from this [link](https://doi.org/10.6084/m9.figshare.29643176.v1). 

## Evaluation

### Training
- Run the following commands for training:
```
python basicsr/train.py -opt options/train/train_GPSMamba_IRSR_x2.yml
```
```
python basicsr/train.py -opt options/train/train_GPSMamba_IRSR_x4.yml
```

### Testing
***
Run 
```
  python basicsr/test.py -opt options/test/GPSMamba/test_GPSMamba_x4.yml
```
```
  python basicsr/test.py -opt options/test/GPSMamba/test_GPSMamba_x2.yml
```
## Setup
For environment setup, you can also refer to these projects:

-  BasicSR-based: [IRSRMamba](https://github.com/yongsongH/IRSRMamba)
-  Mamba-based: [MambaIR](https://github.com/csguoh/MambaIR)

Thanks for their awesome work.

## Contact

If you meet problems, please describe them and [contact](https://hyongsong.work/) me. 

**Impolite or anonymous emails are not welcome. There may be some difficulties for me to respond to the email without self-introduce. Thank you for understanding.**

## Acknowledgement
This work is under peer review.
The updated manuscript and dataset will be released after the paper is accepted.
