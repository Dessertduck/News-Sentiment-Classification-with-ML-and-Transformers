# News Classification Benchmark: From Traditional ML to Transformers

This project presents a comprehensive comparison of traditional machine learning, deep learning, and Transformer-based models for news classification across five categories: Business, Entertainment, Politics, Sports, and Technology. We evaluate performance across multiple architectures, with our best result achieving **98.7% accuracy** using **DistilBERT**.

## 📌 Highlights

- Achieves **98.7% accuracy** with DistilBERT
- Includes traditional ML models (Logistic Regression, XGBoost)
- Implements deep learning models (RNN, LSTM with residual)
- Full technical documentation and expert-annotated predictions
- Supports model export and reproducibility

## 📁 Project Structure

### 🔧 Key Components

#### 1. Core Implementation

| File              | Purpose                                         | Outputs Generated                          |
|-------------------|--------------------------------------------------|---------------------------------------------|
| ML&DL.py  | Trains Logistic Regression, XGBoost, RNN/LSTM   | models/xgb_model.json, FNN_best_params.json |
| BERT.py   | Fine-tunes DistilBERT model                     | models/BERT_best_params.json                |
| Pred.py   | End-to-end prediction pipeline                  | Generates category labels for new inputs    |

#### 2. Data training Resources and data final challenge file

- `data/news-dataset.csv`: 6,000 labeled news articles (train/val split)
- `data/news-challenge.csv`: 1,000 unlabeled articles for final testing

#### 3. Documentation

- `report.pdf`: Full technical paper (methodology, experiments, conclusions)
- `annotated_outcome.pdf`: Model predictions with expert annotations on test set

## 📊 Benchmark Results

| Model           | Accuracy | F1 Score | Training Time | Hardware   |
|----------------|----------|----------|----------------|------------|
| XGBoost         | 93.0%    | 0.930    | 2 min          | CPU-only   |
| LSTM (Residual) | 96.0%    | 0.960    | 35 min         | 1× GPU     |
| **DistilBERT**  | **98.7%**| **0.987**| 68 min         | 1× GPU     |

For detailed predictions and expert comments, see `annotated_outcome.pdf`.

## 🧠 Model Analysis

- **Traditional models** are fast and lightweight, ideal for CPU-only environments.
- **LSTM** captures sequential context and outperforms traditional models.
- **DistilBERT** delivers the best performance due to deep semantic understanding but requires more compute and time.
- All experiments are reproducible with fixed seeds and saved hyperparameters.

## ⚙️ Advanced Features

- Custom text preprocessing rules via `text_cleaner.py`
- Export models in ONNX or PyTorch for deployment
- All configurations and seeds are locked for reproducibility
