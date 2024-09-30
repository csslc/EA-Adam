      
## Perception-Distortion Balanced Super-Resolution: A Multi-Objective Optimization Perspective


<a href='https://arxiv.org/abs/2312.15408'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

[Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2</sup>
| [Jie Liang](https://scholar.google.com/citations?hl=zh-CN&user=REWxLZsAAAAJ)<sup>2</sup> | 
[Shuaizheng Liu](https://scholar.google.com/citations?hl=zh-CN&user=wzdCc-QAAAAJ)<sup>1,2</sup> | 
[Hongwei Yong](https://scholar.google.com.hk/citations?user=Xii74qQAAAAJ&hl=zh-CN)<sup>1</sup> | 
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

:star: If EA-Adam is helpful to your images or projects, please help star this repo. Thanks! :hugs:

## 🌟 Overview framework
![ea-adam](figs/framework.png)

## 🌟 Visual Results
<details>
<summary>SRResNet Backbone (click to expand)</summary>
<p align="center">
<img width="900" src="figs/compare_srresnet.png">
</p>
</details>
<details>
<summary>RRDB Backbone (click to expand)</summary>
<p align="center">
<img width="900" src="figs/compare_rrdb.png">
</p>
</details>
<details>
<summary>SwinIR Backbone (click to expand)</summary>
<p align="center">
<img width="900" src="figs/compare_swinir.png">
</p>
</details>
For more comparisons, please refer to our paper for details.

## ⚙ Dependencies and Installation
```shell
## git clone this repository
git clone https://github.com/csslc/EA-Adam.git
cd EA-Adam

# create an environment
conda create -n EA-Adam python=3.10 -y
conda activate EA-Adam
pip install -r requirements.txt
```

## 🍭 Quick Inference
#### Step 1: Download the pretrained models

Download from [GoogleDrive](https://drive.google.com/drive/folders/15TQWATm6l4iPzAHcOIpOi3Bqvwqv28zz?usp=sharing).

Download from [BaiduNetdisk](https://pan.baidu.com/s/1PdzjTN-eS8zZ-wg5J3GGPQ) 
(pwd: 0930).

#### Step 2: Prepare testing data
You can put the testing images in the `test_input`.

#### Step 3: Running testing command
```
python test.py \
--input_image test_input \
--config configs/mulsrresnet_gan.yml \
--test_model_path pretrained_models/EA-Adam-srresnet.pt \
--root_img output
```
If you want to test RRDB-based and SwinIR-based models, please modify the `test_model_path` and `config` accordingly.

## 🚋 Train
We take the SRResNet backbone as example. Please check and adapt the config files firstly.

1. EA-Adam stage.

A model pretrained with L1 loss needs to be used for stable training, similar to SRGAN, ESRGAN, and other GAN-based SR models.
```
python train_EA-Adam_srresnet.py \
--config configs/mulsrresnet_gan.yml \
--resume l1-pretrained/ \
```
2. Weight regression network training.

`N` expert models can be obtained during the EA-Adam stage and should be placed in `fusion_experts` for training weight regression network.
The final discriminator from the EA-Adam stage can serve as a pretrained model to facilitate faster convergence.

```
python train_fusion_srresnet.py \
--config configs/fusion_srresnet.yml \
--expert_path fusion_experts/ \
--resume_d disc.pt
```

3. Model fusion.

The fusion models from weight regression network training stage is placed in `fusion_model`.
The final model is saved in `final_model_path`.

```
python train_cal_weight.py \
--config configs/fusion_srresnet.yml \
--test_model_path fusion_model/ \
--expert_path fusion_experts/ \
--final_model_path experiments/EA-Adam-srresnet/ 
```

### License
This project is released under the [Apache 2.0 license](LICENSE).

### Acknowledgement
This project is built based on the [SimpleIR](https://github.com/xindongzhang/SimpleIR) and [E-GAN](https://github.com/WANG-Chaoyue/EvolutionaryGAN) projects. Thanks for their awesome works. 

### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@article{sun2024eaadam,
  title={Perception-Distortion Balanced Super-Resolution: A Multi-Objective Optimization Perspective},
  author={Sun, Lingchen and Liang, Jie and Liu, Shuaizheng and Yong, Hongwei and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```

### Contact
If you have any questions, please feel free to contact: ling-chen.sun@connect.polyu.hk


<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=csslc/EA-Adam)

</details>


