🚀 Emotion Detection from Text (Machine Learning Model)

A machine learning–based Emotion Detection System that analyzes text and classifies it into 8 emotion categories:
Joy, Sadness, Fear, Anger, Surprise, Neutral, Disgust, Shame

This project includes a full ML pipeline — preprocessing, model training, evaluation, and model serialization for production use.

📌 Features

🔍 Detects 8 human emotions from raw text

🧹 End-to-end text preprocessing pipeline

⚙️ Trained multiple ML models (LR, SVM, RF)

⭐ Final model achieves 62% accuracy

📦 Exported model via joblib

🚀 Ready for deployment (FastAPI / Streamlit / Flask)

📂 Dataset Summary
Emotion	Count
Joy	11045
Sadness	6722
Fear	5410
Anger	4297
Surprise	4062
Neutral	2254
Disgust	856
Shame	146

⚠️ The dataset is imbalanced, making rare emotions harder to classify.

🧹 Text Preprocessing Pipeline

✔ Remove @mentions
✔ Remove stopwords
✔ Lowercasing
✔ Remove special characters
✔ Tokenization
✔ Convert text → vectors using CountVectorizer

🤖 Machine Learning Models Used
1️⃣ Logistic Regression (Final Model)

Accuracy: 62%

Fastest

Most interpretable

Best balance of speed/performance

2️⃣ Support Vector Machine (RBF Kernel)

Accuracy: 62.2%

High computational cost

Sensitive to hyperparameters

3️⃣ Random Forest

Accuracy: 56.32%

Struggled with sparse text features

🏆 Model Selection Reasoning

Logistic Regression was chosen because:

⚡ Fastest training & inference

📊 Produces interpretable coefficients

🔁 Highly scalable and lightweight

💯 Competitive accuracy

📦 Technologies & Tools
Languages & Frameworks

Python

Scikit-Learn

NLTK / spaCy

Model Management

joblib (model saving)

MLflow / Weights & Biases (optional)

Optional Enhancements

HuggingFace Transformers (BERT, DistilBERT)

Word2Vec / GloVe

TensorFlow Lite / ONNX

🛠️ Project Structure (Recommended)
emotion-detection/
│
├── data/
├── models/
│   └── final_model.joblib
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── requirements.txt
└── README.md

⚠️ Challenges

Extreme class imbalance

Slang, emojis, abbreviations

Sparse features reduce model accuracy

Rare emotions like shame are hard to predict

💡 Future Improvements

Apply SMOTE or class weighting

Replace CountVectorizer with TF-IDF

Use BERT / DistilBERT for better context

Emoji → emotion mapping

Deploy via FastAPI or Streamlit

Add SHAP/LIME for explainability

🚀 Deployment Ready

This model is compatible with:

🔹 FastAPI REST API
uvicorn app:app --reload

🔹 Streamlit Web UI
streamlit run app.py

🔹 Docker
docker build -t emotion-detector .
docker run -p 8000:8000 emotion-detector

📝 Example Usage
from joblib import load

model = load("models/final_model.joblib")
vectorizer = load("models/vectorizer.joblib")

text = ["I am feeling great today!"]

X = vectorizer.transform(text)
prediction = model.predict(X)

print(prediction[0])

📜 License

This project is released under the MIT License.

⭐ Support

If you like this project, consider giving it a star ⭐ on GitHub!