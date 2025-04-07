# News Classification Benchmark: From Traditional ML to Transformers

This project presents a comprehensive comparison of traditional machine learning, deep learning, and Transformer-based models for news classification across five categories: Business, Entertainment, Politics, Sports, and Technology. We evaluate performance across multiple architectures, with our best result achieving **98.7% accuracy** using **DistilBERT**.

## 📌 Highlights

- Achieves **98.7% accuracy** with DistilBERT
- Includes traditional ML models (Logistic Regression, XGBoost)
- Implements deep learning models (RNN, LSTM with residual)
- Full technical documentation and expert-annotated predictions
- Supports model export and reproducibility

## 📁 Project Structure

data/ │ news-dataset.csv # 6,000 labeled articles (train/val) │ news-challenge.csv # 1,000 unlabeled articles (test) models/ │ xgb_model.json # Trained XGBoost model │ FNN_best_params.json # Best parameters for feedforward model │ BERT_best_params.json # Best parameters for DistilBERT scripts/ │ ML&DL.py # Trains traditional and deep learning models │ BERT.py # Fine-tunes DistilBERT │ Pred.py # End-to-end prediction pipeline report.pdf # Full technical report (methods, experiments) annotated_outcome.pdf # Expert-reviewed model predictions on test set

shell
复制
编辑

## 🚀 Getting Started

### Environment Setup

```bash
conda env create -f environment.yml
Training Models
1. Traditional & Deep Learning Models
bash
复制
编辑
python scripts/ML&DL.py --data data/news-dataset.csv --optimize
2. Transformer-based Model (DistilBERT)
bash
复制
编辑
python scripts/BERT.py --data data/news-dataset.csv --epochs 10
Run Predictions
bash
复制
编辑
python scripts/Pred.py --model bert --input data/news-challenge.csv
📊 Benchmark Results
Model	Accuracy	F1 Score	Training Time	Hardware
XGBoost	93.0%	0.930	2 min	CPU-only
LSTM (Residual)	96.0%	0.960	35 min	1× GPU
DistilBERT	98.7%	0.987	68 min	1× GPU
For detailed predictions and expert comments, see annotated_outcome.pdf.

