# [GPSMamba](https://arxiv.org/pdf/2507.18998)
Official PyTorch implementation of the paper [GPSMamba](https://arxiv.org/pdf/2507.18998)


## Introduction

 Infrared Image Super-Resolution (IRSR) is challenged by low-contrast, sparsely-textured data that demands robust global context modeling. While State-Space Models (SSMs) like Mamba offer efficient long-range dependency modeling, their intrinsic 1D causal scanning mechanism fragments the 2D image context, fundamentally limiting reconstruction fidelity. To address this, we present GPSMamba, a framework that systematically overcomes this limitation through a synergy of non-causal architectural prompting and global frequency-domain supervision. First, our Adaptive Semantic-Frequency State Space Module (ASF-SSM) embodies the non-causal prompting. It injects a dynamic prompt, derived from global frequency information, directly into the Mamba block. This breaks the rigid causal chain from within, enabling a non-causal, globally-aware state transition. Second, our Thermal-Spectral Attention and Phase Consistency (TSAPC) Loss provides explicit global supervision. Its thermal-spectral attention mechanism overcomes the limitations of patch-based scanning by selectively enforcing spectral fidelity on sparse thermal targets across non-adjacent patches. Through the synergy of these two innovations, extensive experimental results demonstrate that GPSMamba achieves state-of-the-art performance on benchmark IRSR datasets.
 
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

---

## Legal Action and Patent Clause
This project is licensed under the **Apache License, Version 2.0**.
A copy of the Apache License, Version 2.0, is included in the `LICENSE` file in this repository. You may also obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

**Any violation of the terms of the Apache License, Version 2.0, may result in legal action and liability for damages.**
