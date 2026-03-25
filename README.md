Fake News Detection using Machine Learning with Confidence Scoring
Project Overview

This project presents a Machine Learning-based Fake News Classification system that identifies whether a news article is Real or Fake using Natural Language Processing techniques. The system applies TF-IDF feature extraction and Logistic Regression for classification. Additionally, probability calibration is used to generate confidence scores, and LIME (Explainable AI) highlights important words influencing predictions.

The application is implemented using Flask web framework and provides an interactive interface where users can input news text and view prediction results along with explanation and confidence levels.
Key Features

• Fake news classification using Machine Learning
• TF-IDF vectorization for text feature extraction
• Logistic Regression model for prediction
• Confidence score using probability calibration
• LIME explanation to interpret model decisions
• Text preprocessing using NLP techniques
• Flask-based web interface
• Prediction history tracking

Technologies Used

Programming Language:

Python

Machine Learning:

scikit-learn
TF-IDF Vectorizer
Logistic Regression
CalibratedClassifierCV

Natural Language Processing:

NLTK
Tokenization
Stopword Removal
Lemmatization

Explainable AI:

LIME (Local Interpretable Model-agnostic Explanations)

Web Framework:

Flask

Libraries:

pandas
numpy
joblib
regex

Frontend:
HTML
CSS
Bootstrap

project structure 

fake-news-detection-ml
│
├── app_v2.py
├── train_model_v2.py
├── requirements.txt
├── runtime.txt
├── Procfile
│
├── templates/
│     index.html
│     result.html
│
├── static/
│     css
│     js
│
└── dataset/
      Fake.csv
      True.csv
      fake_or_real_news.csv
      extra_data.csv

Dataset Instructions

Due to GitHub file size limitations, dataset files are not included in this repository.

Download the dataset files and place them inside a folder named:

dataset

Required dataset files:

Fake.csv
True.csv
fake_or_real_news.csv
extra_data.csv

After placing dataset files, run:

python train_model_v2.py

This will generate trained model files:

my_model.pkl
my_vectorizer.pkl
Installation and Setup

Step 1: Clone the repository

git clone https://github.com/ashie53bora/fake-news-detection-ml.git

Step 2: Navigate to project folder

cd fake-news-detection-ml

Step 3: Install required libraries

pip install -r requirements.txt

Step 4: Train the model

python train_model_v2.py

Step 5: Run the application

python app_v2.py

Step 6: Open browser

http://127.0.0.1:3000
Working Process
User enters news text through web interface
Text preprocessing is applied using NLP techniques
TF-IDF converts text into numerical features
Logistic Regression model predicts Real or Fake
Confidence score is calculated using probability calibration
LIME identifies important words influencing prediction
Result displayed in web interface
Output

The system provides:

• Prediction result (Real or Fake)
• Confidence score percentage
• Important words influencing prediction
• Explanation of model decision

Future Enhancements

• Integration of live news API
• Deep learning models for improved accuracy
• Multilingual fake news detection
• Browser extension for automatic detection
• Real-time news verification system

Author

Machine Learning Project
Fake News Detection System

HTML
CSS
Bootstrap
