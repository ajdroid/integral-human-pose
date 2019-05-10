# Kinematic Human Pose Regression
This repository implements kinematic-model based 3D pose regression in Pytorch.
It is based off of [Integral Pose Regression](https://github.com/JimmySuen/integral-human-pose).

## Environment
Python Version: 3.6 <br/>
OS: CentOs7 (Other Linux system is also OK) <br/>
CUDA: 9.0 (least 8.0) <br/>
PyTorch:0.4.0(see issue https://github.com/JimmySuen/integral-human-pose/issues/4)


## Installation
We recommend installing python from [Anaconda](https://www.anaconda.com/), installing pytorch following guide on [PyTorch](https://pytorch.org/) according to your specific CUDA & python version.
In addition, you need to install dependencies below.
```
pip install scipy
pip install matplotlib
pip install opencv-python
pip install easydict
pip install pyyaml
``` 


## Preparation for Training & Testing
1. Download Human3.6M(ECCV18 Challenge) image from [Human3.6M Dataset](http://vision.imar.ro/human3.6m/description.php) and our processed annotation from [Baidu Disk](https://pan.baidu.com/s/1Qg4dH8PBXm8SzApI-uu0GA) (code: kfsm) or [Google Drive](https://drive.google.com/file/d/1wZynXUq91yECVRTFV8Tetvo271BXzxwI/view?usp=sharing)
2. Download MPII image from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/)
3. Download COCO2017 image from [COCO Dataset](http://cocodataset.org/#home)
4. Download cache file from [Dropbox](https://www.dropbox.com/sh/uouev0a1ao84ofd/AADAjJUdr_Fm-eubk7c_s2JTa?dl=0)
5. Organize data like this
```
${PROJECT_ROOT}
 `-- data
     `-- coco
        |-- images
        |-- annotations
        |-- COCO_train2017_cache
     `-- mpii
        |-- images
        |-- annot
        |-- mpii_train_cache
        |-- mpii_valid_cache
     `-- hm36
        |-- images
        |-- annot
        |-- HM36_train_cache
        |-- HM36_validmin_cache
     `-- hm36_eccv_challenge
        `-- Train
            |-- IMG
            |-- POSE
        `-- Val
            |-- IMG
            |-- POSE
        `-- Test
            |-- IMG
        |-- HM36_eccv_challenge_Train_cache
        |-- HM36_eccv_challenge_Test_cache
        |-- HM36_eccv_challenge_Val_cache
```

## Usage
We have placed some example config files in *experiments* folder, and you can use them straight forward. Don't modify them unless you know exactly what it means.
### Train 
For [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229), cd to *pytorch_projects/integral_human_pose* <br/>
**Kinematic Regression**
```bash
python3 train.py --cfg=experiments/hm36/resnet50v1_ft/kinematic_d-mh_ps-256/lr1e-3.yaml --da
taroot="../../data/"
```
**Direct Joint Regression**
```bash
python train.py --cfg=experiments/hm36/resnet50v1_ft/d-mh_ps-256_dj_l1_adam_bs32-4gpus_x140-90-120/lr1e-3.yaml --dataroot=../../data/
```

By default, logging and model will be saved to *log* and *output* folder respectively.

## Misc
- The project is built on old version of pytorch(0.4.0), and currently the latest released one has updated to 1.0.1. So there may be some compatibility problems. Please feel free to submit new issues.
