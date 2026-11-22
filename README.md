# Clothes classification (repo in process, next week with some updates)
Fashion Product Images (Small) is a Kaggle dataset of about 44,000 color product photos from an Indian fashion e-commerce site. Each image comes with a row in styles.csv holding catalogue attributes such as gender, category, article type, color, season and usage. This project uses the images to predict three attributes of each product, comparing classical models against convolutional neural networks.

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Installation
With the virtual environment active, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

On Windows, this installs PyTorch **without GPU support**, because that is the default build on PyPI. With an NVIDIA GPU, install the CUDA build first and then run the command above:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

To check that PyTorch sees the GPU (it should print `True`):

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Targets
The models predict three columns of styles.csv:

| Column | What it is | Classes |
|---|---|---|
| `subCategory` | Product group, such as Topwear, Shoes or Watches | 45 |
| `articleType` | Specific product, such as Tshirts, Casual Shoes or Handbags | 143 |
| `baseColour` | Main colour of the product | 47 |

## Data split
`splitData.py` copies the images into `dataset/train` (70%), `dataset/validate` (20%) and `dataset/test` (10%). Each folder gets its own `styles.csv`. The split is stratified by `articleType`, which almost always determines `subCategory`, so both keep the same class proportions in every folder. It uses a fixed seed, so running it again gives the same split.
