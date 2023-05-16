# HHGR
# Requirements
Dependencies (with python >= 3.7):
```
numpy==1.21.2
torch==1.10.1
torch-cluster==1.5.9                  
torch-geometric==2.0.4                  
torch-scatter==2.0.9                   
torch-sparse==0.6.13                
```

# Preprocessing
## Dataset
Create a folder [data](https://drive.google.com/drive/folders/1azfYzx3dxYUm-ZklZR9SDCs88UZBgkjB?usp=share_link) to store source data files.

## Prepocess the data
Generate the postive samples:
```
python pos.py
```
Generate the hyperedges by using ``hyperedge.ipynb``.


# Training
training with default configuration:
```
# on DBLP dataset
python main_hyper.py --dataset DBLP

# on ACM dataset
python main_hyper.py --dataset ACM

# on Yelp dataset
python main_hyper.py --dataset Yelp
```
