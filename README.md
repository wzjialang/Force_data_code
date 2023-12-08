# A Deep Learning Approach to Classify Surgical Skill in Microsurgery Using Force Data from a Novel Sensorised Surgical Glove
This repository provides the official PyTorch implementation of the following paper:
> [**A Deep Learning Approach to Classify Surgical Skill in Microsurgery Using Force Data from a Novel Sensorised Surgical Glove**](https://doi.org/10.3390/s23218947)<br>
> [Jialang Xu*](https://www.researchgate.net/profile/Jialang-Xu), Dimitrios Anastasiou*, James Booker, Oliver E. Burton, Hugo Layard Horsfall, Carmen Salvadores Fernandez, Yang Xue, Danail Stoyanov, Manish K. Tiwari, Hani J. Marcus and Evangelos B. Mazomenos<br>
\* Co-first authors.

## Introduction
Microsurgery serves as the foundation for numerous operative procedures. Given its highly technical nature, the assessment of surgical skill becomes an essential component of clinical practice and microsurgery education. The interaction forces between surgical tools and tissues play a pivotal role in surgical success, making them a valuable indicator of surgical skill. In this study, we employ six distinct deep learning architectures (LSTM, GRU, Bi-LSTM, CLDNN, TCN, Transformer) specifically designed for the classification of surgical skill levels. We use force data obtained from a novel sensorized surgical glove utilized during a microsurgical task. To enhance the performance of our models, we propose six data augmentation techniques. The proposed frameworks are accompanied by a comprehensive analysis, both quantitative and qualitative, including experiments conducted with two cross-validation schemes and interpretable visualizations of the network’s decision-making process. Our experimental results show that CLDNN and TCN are the top-performing models, achieving impressive accuracy rates of 96.16% and 97.45%, respectively. This not only underscores the effectiveness of our proposed architectures, but also serves as compelling evidence that the force data obtained through the sensorized surgical glove contains valuable information regarding surgical skill.

## Content
### Architecture
<img src="https://github.com/wzjialang/SR-AQA/blob/main/figure/framework_simple.png" height="500"/>


### Dataset
The force dataset published in our paper could be downloaded [here](https://doi.org/10.5522/04/24476641).

### Setup & Usage for the Code
1. Unzip the dowloaded force dataset and check the structure of data folders:
```
(root folder)
├── all_attempts
|  ├── expert_X_attemp_X.csv
|  ├── novice_X_attemp_X.csv
|  ├── ...
```

2. Check dependencies:
```
- Python 3.8+
- PyTorch 1.10+
- cudatoolkit
- cudnn
- tlib
```

- Simple example of dependency installation:
```
conda create -n sraqa python=3.8
conda activate sraqa
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/thuml/Transfer-Learning-Library.git
cd Transfer-Learning-Library/
python setup.py install
pip install -r requirements.txt
```

3. Train & test models:
- For CP task
```
python main_ours.py /path/to/your/dataset \
        -d TEE -s S -t R --task_type cp_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/SR-AQA/TEE_cp --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10
```
- For GI task
```
python main_ours.py /path/to/your/dataset \
        -d TEE -s S -t R --task_type gi_reg --epochs 100 -i 400 --gpu_id cuda:0 --lr 0.0001 \
        -b 32 --log logs/SR-AQA/TEE_gi --resize-size 224 --fs_layer 1 1 1 0 0 --lambda_scl 1 --lambda_tl 1 --t_data_ratio 10
```

*'--fs_layer 1 1 1 0 0'* means replacing the $1- 3^{rd}$ batch normalization layers of ResNet-50 with the UFS.<br>
*'--lambda_scl'* means the lambda for SCL loss, if *'--lambda_scl'* > 0, then using SCL loss.<br>
*'--lambda_tl'* means the lambda for TL loss, if *'--lambda_tl'* > 0, then using TL loss.<br>
*'--t_data_ratio X'* means using X-tenths of unlabeled real data for training.

## Acknowledge
Appreciate the work from the following repositories:
* [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
* [suhyeonlee/WildNet](https://github.com/suhyeonlee/WildNet)

## Cite
If this repository is useful for your research, please cite:
```
@InProceedings{10.1007/978-3-031-43996-4_15,
author="Xu, Jialang
and Jin, Yueming
and Martin, Bruce
and Smith, Andrew
and Wright, Susan
and Stoyanov, Danail
and Mazomenos, Evangelos B.",
editor="Greenspan, Hayit
and Madabhushi, Anant
and Mousavi, Parvin
and Salcudean, Septimiu
and Duncan, James
and Syeda-Mahmood, Tanveer
and Taylor, Russell",
title="Regressing Simulation to Real: Unsupervised Domain Adaptation for Automated Quality Assessment in Transoesophageal Echocardiography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="154--164",
isbn="978-3-031-43996-4"
}
```
