# News Sentiment Classification with ML and Transformers

This project implements a **multi-class sentiment classification system** for news articles, categorizing text into five topics: Business, Entertainment, Politics, Sports, and Technology. It compares traditional machine learning (Logistic Regression, XGBoost), deep learning (RNN, LSTM), and Transformer-based (DistilBERT) models, with DistilBERT achieving the highest accuracy (98.7%). The pipeline includes data preprocessing, feature engineering, hyperparameter tuning, and performance evaluation.

## Key Features
- **Multi-Model Comparison**: Evaluates Logistic Regression, XGBoost, FNN, RNN, LSTM, and DistilBERT.
- **Advanced Techniques**:  
  - TF-IDF and Word2Vec for feature engineering.  
  - Optuna for hyperparameter optimization.  
  - Layer normalization and residual connections in LSTM.  
- **Business-Ready**: Focuses on deployability and computational trade-offs (e.g., DistilBERT vs. Logistic Regression).

## Results
| Model          | Weighted F1 | Accuracy |
|----------------|-------------|----------|
| Logistic Regression | 0.963       | 0.963    |
| LSTM           | 0.960       | 0.960    |
| **DistilBERT**     | **0.987**   | **0.987**|

## Repository Structure

├── data/
│   ├── news-dataset.csv       # Labeled training data (5 categories)
│   └── news-challenge.csv     # Test data for predictions
│
├── models/
│   ├── xgb_model.json         # Serialized XGBoost model
│   ├── BERT_best_params.json  # Optimal parameters for BERT
│   └── FNN_best_params.json   # Optimal parameters for Feedforward NN
│
├── scripts/
│   ├── ML&DL.py      # Traditional ML/DL model training
│   ├── BERT.py       # BERT fine-tuning and evaluation  
│   └── Pred.py          # End-to-end prediction pipeline
│
└── README.md                  # This documentation
