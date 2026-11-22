# Clothes classification (repo in process, next week with some updates)
Fashion Product Images (Small) is a Kaggle dataset of about 44,000 color product photos from an Indian fashion e-commerce site. Each image comes with a row in styles.csv holding catalogue attributes such as gender, category, article type, color, season and usage. This project uses the images to predict three attributes of each product, comparing classical models against convolutional neural networks.

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Targets
The models predict three columns of styles.csv:

| Column | What it is | Classes |
|---|---|---|
| `subCategory` | Product group, such as Topwear, Shoes or Watches | 45 |
| `articleType` | Specific product, such as Tshirts, Casual Shoes or Handbags | 143 |
| `baseColour` | Main colour of the product | 47 |

## Data split
`splitData.py` copies the images into `dataset/train` (70%), `dataset/validate` (20%) and `dataset/test` (10%). Each folder gets its own `styles.csv`. The split is stratified by `articleType`, which almost always determines `subCategory`, so both keep the same class proportions in every folder. It uses a fixed seed, so running it again gives the same split.
