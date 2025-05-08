import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Title of the app
st.title("Diabetes Detection App")

# Load the diabetes.csv file 
try:
    df = pd.read_csv("diabetes.csv")
    st.success("diabetes.csv loaded successfully ")
except FileNotFoundError:
    st.error("Error: diabetes.csv file not found!")
    st.stop()

# Data Exploration 
st.subheader("Dataset Overview")
st.dataframe(df.head())

# Data cleaning
columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_to_clean:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Display descriptive statistics
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
st.subheader("Choose a Machine Learning Model")
model_choice = st.selectbox("Model:", [
    "Random Forest",
    "SVM",
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gradient Boosting"
])

# Initialize model based on user choice
if model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "SVM":
    model = SVC(probability=True)
elif model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "K-Nearest Neighbors":
    model = KNeighborsClassifier()
elif model_choice == "Gradient Boosting":
    model = GradientBoostingClassifier()

# Train the model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
st.subheader("Model Evaluation")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy: {accuracy:.2f}**")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.dataframe(pd.DataFrame(cm, columns=["Non-Diabetic", "Diabetic"], index=["Non-Diabetic", "Diabetic"]))

# ROC Curve
y_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
st.pyplot(fig2)

# Save the model
st.subheader("Save Trained Model")
save_model_button = st.button("Save Trained Model")

if save_model_button:
    model_filename = f"{model_choice}_model.pkl"
    joblib.dump(model, model_filename)
    st.success(f"Model saved as {model_filename}")

# Custom Prediction
st.subheader("Make a Custom Prediction")
user_input = []
for feature in X.columns:
    value = st.number_input(f"Value for {feature}", min_value=0.0)
    user_input.append(value)

if st.button("Predict"):
    user_scaled = scaler.transform([user_input])
    prediction = model.predict(user_scaled)
    st.success(f"Prediction result: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")

# RAG

from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Initialize Streamlit UI
st.markdown("---")
st.header(" Diabetes Question Answering (RAG with BERT)")

# Installation check
try:
    # Load diabetes knowledge base
    with open("diabetes_knowledge.txt", "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]
    st.success(" Knowledge base loaded successfully!")
except FileNotFoundError:
    st.error(" diabetes_knowledge.txt not found! Please ensure the file exists.")
    st.stop()

# Initialize models
@st.cache_resource
def load_models():
    try:
        # Embedding model
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # BERT QA model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
        
        return embedder, tokenizer, model
    except Exception as e:
        st.error(f" Model loading failed: {str(e)}")
        st.stop()

embedder, tokenizer, qa_model = load_models()

# Create FAISS index
try:
    passage_embeddings = embedder.encode(passages, convert_to_tensor=False)
    index = faiss.IndexFlatL2(passage_embeddings.shape[1])
    index.add(passage_embeddings)
    st.session_state['index'] = index
except Exception as e:
    st.error(f" Error building search index: {str(e)}")
    st.stop()

# User input
question = st.text_input(
    "Ask about diabetes:",
    placeholder="e.g., What are the symptoms of type 2 diabetes?"
)

if st.button("Get Answer", type="primary"):
    if not question:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🔍 Analyzing your question..."):
            try:
                # 1. Retrieve relevant passages
                question_embedding = embedder.encode([question])[0]
                _, I = st.session_state['index'].search(
                    np.array([question_embedding]), 
                    k=3
                )
                contexts = [passages[i] for i in I[0]]

                # 2. Get BERT answers for each context
                answers = []
                for context in contexts:
                    inputs = tokenizer(
                        question, 
                        context, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = qa_model(**inputs)
                    
                    answer_start = torch.argmax(outputs.start_logits)
                    answer_end = torch.argmax(outputs.end_logits) + 1
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0][answer_start:answer_end]
                        )
                    )
                    
                    # Calculate confidence score
                    score = (outputs.start_logits[0][answer_start] + 
                            outputs.end_logits[0][answer_end-1]).item()
                    
                    answers.append({
                        'answer': answer,
                        'score': score,
                        'context': context
                    })

                # 3. Display best answer
                if answers:
                    best_answer = max(answers, key=lambda x: x['score'])
                    
                    st.divider()
                    st.markdown(f"###  Answer:")
                    st.success(best_answer['answer'])
                    
                    st.markdown(f"###  Source Context:")
                    st.text(best_answer['context'])
                    
                    st.markdown(f"###  Confidence: {best_answer['score']:.2f}")
                    
                    # Show other potential answers
                    with st.expander("View alternative answers"):
                        for i, ans in enumerate(answers):
                            if ans != best_answer:
                                st.markdown(f"**Option {i+1}** (Score: {ans['score']:.2f})")
                                st.info(ans['answer'])
                                st.text(f"Context: {ans['context']}")
                                st.divider()
                
            except Exception as e:
                st.error(f" Error processing question: {str(e)}")