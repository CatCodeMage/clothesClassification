# KAGGLE CASE PLANNING

## Stages
1. Preprocessing
2. Training
3. Evaluation
4. Inference (dashboard)

### Preprocessing
Extract features from images and store batches in features/

### Training
Run different models from less to more complex and store them in models/

Models chosen:
- Dummy Classifier (most frequent class)
- Support Vector Machine
- KNN
- K-means
- Neural Networks
- Convolutional Neural Networks

### Evaluation
Store metrics (JSON) and plots, and show them in a local web dashboard.

Metrics:
- Precision
- Recall
- Accuracy
- F1

- Classification report (metrics)

Plots:
- ROC Curve
- ROC_AUC
- Confusion matrix

### Inference
In each model, extract its features and predict. Make inference in web dashboard.