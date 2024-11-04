# <center> EA-multi-head-GCN
Implementation of paper: Building Pattern Recognition by Using an Edge-attention Multi-head Graph Convolutional Network

Github: [eamhgcn](https://github.com/whtower/eamhgcn)

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
```

```
