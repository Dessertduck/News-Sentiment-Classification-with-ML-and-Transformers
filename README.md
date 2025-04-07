News Classification Benchmark: From Traditional ML to Transformers
This repository contains a comprehensive study comparing machine learning approaches for news categorization. Our work evaluates traditional machine learning, deep learning, and Transformer-based models on a dataset spanning five news domains: Business, Entertainment, Politics, Sports, and Technology. Detailed methodology, experiments, and results are documented in our full technical report.

Overview
The project benchmarks multiple models for news categorization:

Traditional ML & Deep Learning Models

Logistic Regression, XGBoost, RNN, and LSTM

Transformer-based Model

DistilBERT

Key findings from our experiments include:

DistilBERT achieved the highest accuracy of 98.7% with strong F1 performance.

LSTM provided competitive results among deep learning methods.

Traditional models such as Logistic Regression and XGBoost were efficient but less effective in handling contextual nuances.

Our experiments also cover detailed aspects such as text preprocessing, feature engineering (including TF-IDF, N-grams, and Word2Vec), hyperparameter optimization using Optuna, and evaluation metrics like weighted F1 score and overall accuracy.
