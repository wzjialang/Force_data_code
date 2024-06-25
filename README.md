# A Deep Learning Approach to Classify Surgical Skill in Microsurgery Using Force Data from a Novel Sensorised Surgical Glove
This repository provides the official PyTorch implementation of the following paper:
> [**A Deep Learning Approach to Classify Surgical Skill in Microsurgery Using Force Data from a Novel Sensorised Surgical Glove**](https://doi.org/10.3390/s23218947)<br>
> [Jialang Xu*](https://www.researchgate.net/profile/Jialang-Xu), Dimitrios Anastasiou*, James Booker, Oliver E. Burton, Hugo Layard Horsfall, Carmen Salvadores Fernandez, Yang Xue, Danail Stoyanov, Manish K. Tiwari, Hani J. Marcus and Evangelos B. Mazomenos<br>
\* Co-first authors.

## Introduction
Microsurgery serves as the foundation for numerous operative procedures. Given its highly technical nature, the assessment of surgical skill becomes an essential component of clinical practice and microsurgery education. The interaction forces between surgical tools and tissues play a pivotal role in surgical success, making them a valuable indicator of surgical skill. In this study, we employ six distinct deep learning architectures (LSTM, GRU, Bi-LSTM, CLDNN, TCN, Transformer) specifically designed for the classification of surgical skill levels. We use force data obtained from a novel sensorized surgical glove utilized during a microsurgical task. To enhance the performance of our models, we propose six data augmentation techniques. The proposed frameworks are accompanied by a comprehensive analysis, both quantitative and qualitative, including experiments conducted with two cross-validation schemes and interpretable visualizations of the network’s decision-making process. Our experimental results show that CLDNN and TCN are the top-performing models, achieving impressive accuracy rates of 96.16% and 97.45%, respectively. This not only underscores the effectiveness of our proposed architectures, but also serves as compelling evidence that the force data obtained through the sensorized surgical glove contains valuable information regarding surgical skill.

## Content
### Results
<img src="https://github.com/wzjialang/Force_data_code/blob/main/figure/Result1-2.png" height="300">
<img src="https://github.com/wzjialang/Force_data_code/blob/main/figure/Result3.png" height="250">
<img src="https://github.com/wzjialang/Force_data_code/blob/main/figure/Result4.png" height="250">


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

3. Train & test models and augmentations:
```
python experiment.py -exp model_name -aug {augmentation_name}
```
*model_name = [gru, lstm, bidrectional, cldnn, simpletcn, transformer].<br>*
*augmentation_name = [fft, drift, quantize, timewarp, gaussian, tst].<br>*

## Cite
If this repository is useful for your research, please cite:
```
@Article{xu2023deep,
AUTHOR = {Xu, Jialang and Anastasiou, Dimitrios and Booker, James and Burton, Oliver E. and Layard Horsfall, Hugo and Salvadores Fernandez, Carmen and Xue, Yang and Stoyanov, Danail and Tiwari, Manish K. and Marcus, Hani J. and Mazomenos, Evangelos B.},
TITLE = {A Deep Learning Approach to Classify Surgical Skill in Microsurgery Using Force Data from a Novel Sensorised Surgical Glove},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {21},
ARTICLE-NUMBER = {8947},
URL = {https://www.mdpi.com/1424-8220/23/21/8947},
PubMedID = {37960645},
ISSN = {1424-8220},
DOI = {10.3390/s23218947}
}
```
