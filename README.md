#  Diabetes Detection & Q&A App

This project is a machine learning and NLP-powered web application built with **Streamlit**. It provides two main functionalities:

1. **Predictive Modeling** – Train and evaluate various machine learning models to predict diabetes based on medical features.
2. **RAG-based Q&A System** – Ask diabetes-related questions and get answers retrieved from a domain-specific knowledge base using **Retrieval-Augmented Generation** with **BERT**.

---

##  Features

###  Machine Learning for Diabetes Prediction
- Train and evaluate multiple machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
- Visualize:
  - Accuracy
  - Confusion Matrix
  - ROC Curve
- Save and load trained models using `joblib`.
- Enter custom input values for predictions.

###  Diabetes Question Answering (RAG + BERT)
- Ask natural language questions about diabetes.
- Uses:
  - `SentenceTransformer` to embed your question and retrieve relevant contexts.
  - `FAISS` to perform fast semantic search over a local knowledge base.
  - `BERT` for extracting precise answers from the retrieved contexts.
- Includes spell correction using `pyspellchecker` for noisy queries.

---
## RUN
pip install -r requirements.txt 
python -m streamlit run app.py
