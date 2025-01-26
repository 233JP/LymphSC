# LymSC

##  Introduction
In this study, we tested different AI models for molecular SMILES to implement this prediction tasks. They are GCN, GAT and Graph Transformer, where GT is in' src/main'.

##  Dependencies

+ cuda >= 9.0
+ cudnn >= 7.0
+ RDKit == 2020.03.4
+ torch >= 1.4.0 (please upgrade your torch version in order to reduce the training time)
+ numpy == 1.19.1
+ scikit-learn == 1.3.0
+ tqdm == 4.52.0
+ transformers == 4.31.0
+ torch-geometric == 1.7.2

You can build a conda environment using [requirements.txt](requirements.txt) except PyTorch like the following commands. For PyTorch package, please install the proper version according to your GPU environment.
```sh
conda create -n lymphSC python==3.8
conda activate lymphSC
conda install pip
pip install -r requirements.txt
```

##  Data preprocess
For the origibal dataset, you can find it in LymSC-main/Data/Data/lymph/source

Please run jupyter notebook under LymSC-main/Data/Data/lymph/preprocessing.ipynb

Note that if you change the experimental baseline, don't forget to change the corresponding `dataset` and `split`! For example:
```sh
>> python main.py \
    --experiment_name test \
    --gpu 0 \
    --fold 1 \
    --dataset lymph \
    --split scaffold \
    --gpu 1 \
```
where `<seed>` is the seed number, `<gpu>` is the gpu index number, `<split>` is the split method (except for qm9 is random, all are scaffold), `<dataset>` is the element name
All hyperparameters can be tuned in the `utils.py`

##  Data Generation





