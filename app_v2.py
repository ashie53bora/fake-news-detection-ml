# =============================================================================
# Fake News Detection Project - Version 2 (app_v2.py)
# Features: Local ML + REST API Endpoints + LIME Explainability
# =============================================================================

from flask import Flask, render_template, request, jsonify
import re
import nltk
import joblib
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ================= 1. CONFIGURATION =================
# Download NLTK data (Run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize Flask (FIXED: __name__)
app = Flask(__name__, template_folder='./templates', static_folder='./static')
app.secret_key = "dev_key"

# In-memory history
prediction_history = []

# ================= 2. LOAD MODEL =================
try:
    loaded_model = joblib.load("my_model.pkl")
    vectorizer = joblib.load("my_vectorizer.pkl")
    print("✅ Model and Vectorizer loaded successfully")
except FileNotFoundError:
    print("❌ ERROR: Model files not found. Please run train_model_v2.py first.")
    exit()

# ================= 3. PREPROCESSING (Must match train_model_v2.py) =================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# CRITICAL: Must match train_model_v2.py exactly
stop_words.update(["reuters", "breaking", "exclusive"])

def preprocess_text(text):
    """Clean text exactly like training script"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ================= 4. ML PREDICTION =================
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

def predict_proba_lime(texts):
    """LIME requires probability predictions"""
    cleaned = [preprocess_text(news) for news in texts]
    vectorized = vectorizer.transform(cleaned)
    return loaded_model.predict_proba(vectorized)

def get_ml_prediction(text):
    """Get prediction from loaded model"""
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    
    prediction = loaded_model.predict(vectorized)[0]
    probabilities = loaded_model.predict_proba(vectorized)[0]
    
    fake_conf = round(probabilities[0] * 100, 2)
    real_conf = round(probabilities[1] * 100, 2)
    
    return prediction, fake_conf, real_conf, cleaned

# ================= 5. ROUTES (HTML) =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    return render_template('history.html', history=prediction_history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        
        # Validate input
        if not message or len(message) < 10:
            return render_template("prediction.html", error="Please enter valid text (min 10 characters)")
        
        # 1. ML Prediction
        pred, fake_conf, real_conf, cleaned_text = get_ml_prediction(message)
        
        # 2. LIME Explanation
        exp = explainer.explain_instance(cleaned_text, predict_proba_lime, num_features=10)
        word_weights = exp.as_list()
        
        # Separate words pushing toward FAKE vs REAL
        fake_words = [word for word, weight in word_weights if weight > 0]
        real_words = [word for word, weight in word_weights if weight < 0]
        
        # 3. Build Explanation Message (FIXED: variable name typo)
        if pred == 0:  # FAKE
            if fake_conf > 80:
                reason = "strongly suggests fabricated or sensationalist content"
            elif fake_conf >= 60:
                reason = "contains patterns commonly found in misleading articles"
            else:
                reason = "has some characteristics of unreliable reporting"
            
            if fake_words:
                explanation_message = (
                    f"The model flagged this as fake because the language {reason}. "
                    f"Words like '{', '.join(fake_words[:3])}' are frequently associated with fake news. "
                )
            else:
                explanation_message = (
                    f"The model flagged this as fake because the overall language {reason}. "
                )
        else:  # REAL
            if real_conf > 80:
                reason = "strongly matches the language patterns of credible journalism"
            elif real_conf >= 60:
                reason = "uses neutral, factual language typical of reliable reporting"
            else:
                reason = "leans toward credible content but with moderate confidence"
            
            if real_words:
                explanation_message = (
                    f"The model rated this as real because the language {reason}. "
                    f"Words like '{', '.join(real_words[:3])}' are commonly found in verified reporting. "
                )
            else:
                explanation_message = (
                    f"The model rated this as real because the overall language {reason}. "
                )
        
        # 4. Label & Severity Mapping
        if pred == 0:
            result = "Prediction: FAKE NEWS 📰"
            confidence = fake_conf
            if fake_conf > 80:
                severity = "High Risk"
            elif fake_conf >= 60:
                severity = "Medium Risk"
            else:
                severity = "Low Risk"
        else:
            result = "Prediction: REAL NEWS 📰"
            confidence = real_conf
            severity = "Safe"
        
        # 5. Save to History
        prediction_history.append({
            'headline': message[:80] + '...' if len(message) > 80 else message,
            'result': 'FAKE' if pred == 0 else 'REAL',
            'fake_conf': fake_conf,
            'real_conf': real_conf,
            'severity': severity
        })
        
        # 6. Render Result
        return render_template(
            "result.html",
            prediction_text=result,
            confidence=confidence,
            severity=severity,
            fake_conf=fake_conf,
            real_conf=real_conf,
            explanation_message=explanation_message
        )
    else:
        return render_template("prediction.html")

# ================= 6. REST API ENDPOINTS (JSON) =================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API Endpoint for predictions
    Accepts: JSON {"text": "news content here"}
    Returns: JSON {"prediction": "FAKE/REAL", "confidence": {...}}
    """
    data = request.get_json()
    
    # Validate Input
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided', 'success': False}), 400
    
    message = data['text']
    
    if len(message) < 10:
        return jsonify({'error': 'Text too short (min 10 characters)', 'success': False}), 400
    
    if len(message) > 5000:
        return jsonify({'error': 'Text too long (max 5000 characters)', 'success': False}), 400
    
    # ML Prediction
    pred, fake_conf, real_conf, _ = get_ml_prediction(message)
    
    # Determine severity
    if pred == 0:
        if fake_conf > 80:
            severity = "High Risk"
        elif fake_conf >= 60:
            severity = "Medium Risk"
        else:
            severity = "Low Risk"
    else:
        severity = "Safe"
    
    # Return JSON Response
    return jsonify({
        'success': True,
        'prediction': 'FAKE' if pred == 0 else 'REAL',
        'confidence': {
            'fake': fake_conf,
            'real': real_conf
        },
        'severity': severity,
        'message': message[:100] + '...' if len(message) > 100 else message
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'model_loaded': True if loaded_model else False,
        'vectorizer_loaded': True if vectorizer else False,
        'version': '2.0',
        'endpoints': ['/api/predict', '/api/health']
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    """Get prediction history via API"""
    return jsonify({
        'success': True,
        'count': len(prediction_history),
        'history': prediction_history[-10:]  # Last 10 predictions
    })

# ================= 7. RUN APP (FIXED: __name__) =================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Fake News Detection API Server Starting...")
    print("="*60)
    print("📍 Web Interface: http://localhost:3000")
    print("📍 API Endpoint:  http://localhost:3000/api/predict")
    print("📍 Health Check:  http://localhost:3000/api/health")
    print("="*60 + "\n")
    
    import os

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)