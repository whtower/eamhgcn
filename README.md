# <div align="center">EA-Multi-Head-GCN</div>
Implementation of paper: Building Pattern Recognition by Using an Edge-attention Multi-head Graph Convolutional Network

[Github](https://github.com/whtower/eamhgcn) • [Figshare](https://doi.org/10.6084/m9.figshare.27602619) • [Paper](https://doi.org/10.1080/13658816.2024.2427853)

## Setup
```bash
conda create -n eamhgcn python=3.10
conda activate eamhgcn
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm scikit-learn
```

## Train
```bash
python CODE/main/multi_train.py --model GCN_A_H_S --model_num 5
```

## Test
```bash
 python CODE/main/test.py --model GCN_A_H_S --model_num 5
```

## Make Dataset[Optional]
If you want to make your own dataset, you can use the following command to make it.
- Install the python packages:
```bash
pip install pyshp matplotlib scipy
```
- Install the package of [shapely](https://github.com/shapely/shapely) by your self.
- Run the code `CODE/predata/main.py` to make the dataset.

## Directory Structure
```
├── [dir]CODE
│   ├── [dir]main
│   │   ├── global config file
│   │   ├── test.py
│   │   └── train.py
│   │   ├── [dir]model
│   │   │   ├── [dir]models
│   │   │   │   ├── config.py
│   │   │   │   ├── graph.py
│   │   │   │   ├── layers.py
│   │   │   │   └── model.py
│   │   ├── [dir]utils
│   │   │   └── global util files
│   ├── predata
│   │   ├── BuildingDataLoader.py
│   │   ├── Skeleton.py
│   │   └── main.py
├── [dir]DATA
│   ├── [dir]dataset
│   └── [dir]split_data #shape file
```

## Citation
If you find this repository useful in your research, please cite our paper:
```bibtex
@article{doi:10.1080/13658816.2024.2427853,
author = {Wang, Haitao and Xu, Yongyang and Hu, Anna and Xie, Xuejing and Chen, Siqiong and Xie, Zhong},
title = {Building pattern recognition by using an edge-attention multi-head graph convolutional network},
journal = {International Journal of Geographical Information Science},
volume = {0},
number = {0},
pages = {1--26},
year = {2024},
publisher = {Taylor \& Francis},
doi = {10.1080/13658816.2024.2427853}}
```
<div align="center">✨ Happy research! ✨</div>