English | [简体中文](README_cn.md)

<div style="text-align: center; margin: 10px">
    <h1> ⭐ TGN: Text-Guided Diverse Image Synthesis for Long-Tailed Remote Sensing Object Classification </h1>
</div>
<p align="center">
    <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=36">
    <img alt="Static Badge" src="https://img.shields.io/badge/TGRS-blue?logo=ieee&labelColor=blue&color=blue">
    </a>
    <a href="https://ieeexplore.ieee.org/document/10582893">
    <img alt="Static Badge" src="https://img.shields.io/badge/Paper-openproject.svg?logo=openproject&color=%23B31B1B">
    </a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win-pink.svg"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/XinR-Tang/TGN">
    <a href="mailto: tanghaojun_cam@163.com">
    <img alt="Static Badge" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

---

## 🌋  Notes
This is the official implementation for our <span style='color: #EB5353;font-weight:bold'>TGRS 2024</span> paper "[Text-Guided Diverse Image Synthesis for Long-Tailed Remote Sensing Object Classification](https://ieeexplore.ieee.org/document/10582893)". 
You can quickly implement our work through this project. If you have any questions, please contact us!

![](./picture/1.png)
![](./picture/2.png)

---

## 💡 Introduction
TGN comprises two main components: knowledge mutual
distillation network (KMDN) and class-consistent diverse tail
class generation network (CDTG). KMDN resolves the isolation
issue of the head and tail knowledge by facilitating mutual
learning of feature representations between the head and tail
data, thereby improving the feature extraction capability of
the tail model. CDTG focuses on generating class-consistency
diverse tail class images that uses tail-class features extracted
by KMDN. Especially, the class consistency is guaranteed by
CLIP’s powerful text-image alignment capability. These generated images are then added back into the original dataset to
alleviate the long-tailed distribution, thereby improving the tail
class accuracy.

![](./picture/3.png)
<div align="center">
  <img src="./picture/4.png" width=500 >
</div>

## 🚀 Quick start

### 📍 Install

```bash
pip install -r requirements.txt
```

### 🏕️ Preparing the dataset

We conduct experiments on three remote sensing datasets: DIOR, FGSC-23 and DOTA.

- `DIOR`: Contains 192,465 images from 20 categories, 68,025 samples for training and 124,440 samples for testing.
- `FGSC`: Contains 4,081 images from 23 categories, 3,256 samples for training and 825 samples for testing.
- `DOTA`: Contains 127759 images from 15 categories, 98906 samples for training and 28853 samples for testing.

You can download the preprocessed datasets from [Datasets](https://pan.baidu.com/s/1HebHIjbNpGO0u4nrqu6Wag?pwd=wo4j); Extract code: `wo4j`

- Make sure your project is structured as follows：

```
  ├── CDTG
  │   ├── checkpoint
  │   ├── lpips
  │   |   ...
  │
  ├── Classification
  │   ├── Datasets.py
  │   ├── model_finetune.py
  │   ├── test.py
  │   ├── train.py
  │   ├── Utils.py
  │ 
  ├── KMDN
  │   ├── dataset_split.py
  │   ├── Datasets.py
  │   ├── Distill.py
  │   ├── ....
  │
  ├── dior
  ├── DOTA
  ├── FGSC-23
  ├── README.md
  ├── requirements.txt
```

- If you want to use your own dataset, make sure the dataset has the same structure as follows:
```
  ├── dior
  │   ├── anno
  │   │   ├── DIOR_train.txt
  │   │   ├── DIOR_test.txt
  │   │
  │   ├── train
  │   │   ├── 0
  │   │   │   ├── 00008_0.jpg
  │   │   │   ├── ...
  │   │   │   
  │   │   ├── 1
  │   │   ├── ...  
  │   │
  │   ├── test
  │   │   ├── 0
  │   │   │   ├── 11726_0.jpg
  │   │   │   ├── ...
  │   │   │   
  │   │   ├── 1
  │   │   ├── ... 
```

- Before start, run the following command to split the head dataset and the tail dataset:
```bash
cd KMDN
python3 dataset_split.py
```

`Note:` dataset_split.py is included in both KMDN and CDTG, but they perform different functions!

### 🔥 Preparing the pre-trained weights
We provide the weights for the training. You can quickly implement our work with these weights.
- `result.pth`: You can use this weight to implement our pre-trained classification network. This weight is trained on a dataset with 7000 generated samples added for each tail class image. [Baidu Netdisk](https://pan.baidu.com/s/1HoACbWxesL8cLWNvAw2EWw?pwd=ii10); Extract code: `ii10`.
- `200000.pt`:  You can use this weight to generate the tail class image. [Baidu Netdisk](https://pan.baidu.com/s/17nQJyromz4ap2lS4KfbQFA?pwd=9h1w); Extract code: `9h1w`.

### 🏕️ Testing

You can quickly reproduce our results with the following command(`result.pth` is placed in "./Classification/save_model/result.pth"):

```bash
cd Classification
python3 test.py
```

## 🦄 Train and Evaluation
### 🔥 KMDN

- First, train the head and tail models separately using the following command:

```bash
cd KMDN
python3 dataset_split.py
python3 UModel.py
python3 RModel.py
```

- Then run the following command to perform knowledge mutual distillation:

```bash
python3 Distill.py
```

### 🔥 CDTG
- The dataset structure required by CDTG is as follows:
```
  ├── dior
  │   ├── train
  │   │   ├── tail
  │   │   │   ├── 0
  │   │   │   │   ├── 00008_0.jpg
  │   │   │   │   ├── ...
  │   │   │   
  │   │   │   ├── 1
  │   │   │   ├── ...  
```

- You can run the following command to automatically split into this structure.

```bash
cd CDTG
python3 dataset_split.py
```

If you need to use your own dataset, make sure it has the same structure.

- Run the following command to train the CDTG:

```bash
python3  train.py  --ckpt  checkpoint/your_model_path 
```

### 🔥 Generation

- Finally, you can run the following command to generate different classes of tail images:

```bash
cd CDTG
python3  generation.py  --ckpt  checkpoint/your_model_path  --folder_number  0  --r  0.05
```


- `folder_number:` The class labels for generated image.
- `r:`  Proportion of noise

_`your_model_path` can be replaced by `200000.pt`_

### 🔥 Evaluation
Use the following command to train a classifier on the expanded dataset:

```bash
cd Classification
python3  train.py
```

## Citation
If you find this work useful in your research, please cite our paper:
```
@article{TGRS2024tgn,
  title={Text-Guided Diverse Image Synthesis for Long-Tailed Remote Sensing Object Classification},
  author={Haojun Tang, Wenda Zhao, Guang Hu, Yi Xiao, Yunlong Li and Haipeng Wang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

## Acknowledgements
- This repository is built upon [SatConcepts](https://github.com/kostagiolasn/SatConcepts). 
- Thank the authors of these open source repositories for their efforts. And thank the ACs and reviewers for their effort when dealing with our paper.
