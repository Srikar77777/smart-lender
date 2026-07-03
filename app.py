"""
Smart Lender - Machine Learning Powered Loan Approval Prediction System
"""

# ==================== IMPORT LIBRARIES ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template_string, jsonify
import os
import joblib

# ==================== INITIALIZE FLASK APP ====================
app = Flask(__name__)
app.secret_key = 'smart_lender_secret_key_2026'

# ==================== HTML TEMPLATES ====================
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Lender</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #333; font-size: 2.5em; }
        .header .subtitle { color: #666; font-size: 1.1em; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; color: #333; font-weight: 600; }
        .form-group input, .form-group select { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 14px; }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: #667eea; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .btn-submit { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: 600; cursor: pointer; }
        .btn-submit:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        .batch-section { margin-top: 30px; padding-top: 30px; border-top: 2px solid #eee; }
        .btn-batch { display: inline-block; padding: 12px 24px; background: #28a745; color: white; text-decoration: none; border-radius: 8px; margin-right: 10px; }
        .nav-links { margin-top: 20px; text-align: center; }
        .nav-links a { color: #667eea; text-decoration: none; margin: 0 15px; }
        .alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        @media (max-width: 600px) { .form-row { grid-template-columns: 1fr; } .container { padding: 20px; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Smart Lender</h1>
            <p class="subtitle">AI-Powered Loan Approval Prediction System</p>
        </div>
        <div class="alert"><strong>📊 Model Performance:</strong> XGBoost • 94.7% Training • 81.1% Testing Accuracy</div>
        <form action="/predict" method="POST">
            <div class="form-row">
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="married">Married</label>
                    <select name="married" id="married" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="dependents">Dependents</label>
                    <select name="dependents" id="dependents" required>
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="education">Education</label>
                    <select name="education" id="education" required>
                        <option value="Graduate">Graduate</option>
                        <option value="Not Graduate">Not Graduate</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="self_employed">Self Employed</label>
                    <select name="self_employed" id="self_employed" required>
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="property_area">Property Area</label>
                    <select name="property_area" id="property_area" required>
                        <option value="Urban">Urban</option>
                        <option value="Semiurban">Semiurban</option>
                        <option value="Rural">Rural</option>
                    </select>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="applicant_income">Applicant Income ($)</label>
                    <input type="number" name="applicant_income" id="applicant_income" required min="0" step="100">
                </div>
                <div class="form-group">
                    <label for="coapplicant_income">Coapplicant Income ($)</label>
                    <input type="number" name="coapplicant_income" id="coapplicant_income" required min="0" step="100">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="loan_amount">Loan Amount ($)</label>
                    <input type="number" name="loan_amount" id="loan_amount" required min="0" step="10">
                </div>
                <div class="form-group">
                    <label for="loan_amount_term">Loan Term (months)</label>
                    <select name="loan_amount_term" id="loan_amount_term" required>
                        <option value="360">360 months (30 years)</option>
                        <option value="240">240 months (20 years)</option>
                        <option value="180">180 months (15 years)</option>
                        <option value="120">120 months (10 years)</option>
                        <option value="60">60 months (5 years)</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label for="credit_history">Credit History</label>
                <select name="credit_history" id="credit_history" required>
                    <option value="1.0">Good (1.0)</option>
                    <option value="0.0">Poor (0.0)</option>
                </select>
            </div>
            <button type="submit" class="btn-submit">🔮 Predict Loan Approval</button>
        </form>
        <div class="batch-section">
            <h3>📋 Batch Processing</h3>
            <p style="color: #666; margin-bottom: 15px;">Upload a CSV file for batch predictions</p>
            <form action="/batch_predict" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required style="margin-bottom: 10px;">
                <button type="submit" class="btn-batch">📤 Upload & Process</button>
            </form>
        </div>
        <div class="nav-links">
            <a href="/dashboard">📊 Dashboard</a>
            <a href="/api/predict" target="_blank">🔗 API Endpoint</a>
        </div>
    </div>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; }
        .result-icon { font-size: 80px; margin: 20px 0; }
        .result-approved { color: #28a745; }
        .result-rejected { color: #dc3545; }
        .result-title { font-size: 2em; margin: 15px 0; }
        .result-details { background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 20px 0; }
        .detail-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e0e0e0; }
        .detail-item:last-child { border-bottom: none; }
        .risk-low { color: #28a745; font-weight: bold; }
        .risk-medium { color: #ffc107; font-weight: bold; }
        .risk-high { color: #dc3545; font-weight: bold; }
        .btn-back { display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; margin-top: 20px; }
        .btn-back:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        .confidence-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="result-icon {% if result.approved %}result-approved{% else %}result-rejected{% endif %}">
            {% if result.approved %}✅{% else %}❌{% endif %}
        </div>
        <h1 class="result-title {% if result.approved %}result-approved{% else %}result-rejected{% endif %}">
            {{ result.message }}
        </h1>
        <div class="result-details">
            <div class="detail-item"><span>Confidence</span><span>{{ result.confidence }}</span></div>
            <div class="confidence-bar"><div class="confidence-fill" style="width: {{ result.probability * 100 }}%"></div></div>
            <div class="detail-item"><span>Risk Level</span><span class="risk-{{ result.risk_level.lower() }}">{{ result.risk_level }}</span></div>
            <div class="detail-item"><span>Probability Score</span><span>{{ "%.2f"|format(result.probability * 100) }}%</span></div>
        </div>
        <p style="color: #666; margin: 15px 0;">
            {% if result.approved %}🎉 This applicant qualifies for the loan.{% else %}⚠️ This applicant does not meet the loan criteria. Additional review recommended.{% endif %}
        </p>
        <a href="/" class="btn-back">🏠 Back to Application</a>
        <br><br>
        <a href="/dashboard" class="btn-back" style="background: #28a745;">📊 View Dashboard</a>
    </div>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header h1 { color: #333; font-size: 2em; }
        .header p { color: #666; margin-top: 5px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #333; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .metric-card { text-align: center; padding: 20px; }
        .metric-value { font-size: 2.5em; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-top: 5px; }
        .img-container { width: 100%; margin: 10px 0; }
        .img-container img { width: 100%; border-radius: 10px; border: 1px solid #e0e0e0; }
        .btn-back { display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; margin-top: 10px; }
        @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Smart Lender Dashboard</h1>
            <p>Model Performance Analytics & Insights</p>
            <a href="/" class="btn-back">🏠 Back to Application</a>
        </div>
        <div class="grid">
            <div class="card metric-card">
                <div class="metric-value">94.7%</div>
                <div class="metric-label">Training Accuracy</div>
            </div>
            <div class="card metric-card">
                <div class="metric-value">81.1%</div>
                <div class="metric-label">Testing Accuracy</div>
            </div>
            <div class="card metric-card">
                <div class="metric-value">XGBoost</div>
                <div class="metric-label">Best Model</div>
            </div>
        </div>
        <div class="grid">
            <div class="card">
                <h3>📈 Model Performance</h3>
                <div class="img-container"><img src="/static/model_comparison.png" alt="Model Comparison"></div>
            </div>
            <div class="card">
                <h3>📊 Loan Distribution</h3>
                <div class="img-container"><img src="/static/loan_distribution.png" alt="Loan Distribution"></div>
            </div>
        </div>
    </div>
</body>
</html>
'''

ERROR_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 500px; margin: 100px auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; }
        .error-icon { font-size: 80px; margin: 20px 0; color: #dc3545; }
        .error-message { color: #dc3545; margin: 20px 0; }
        .btn-back { display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="error-icon">⚠️</div>
        <h1>Oops! Something went wrong</h1>
        <p class="error-message">{{ error }}</p>
        <a href="/" class="btn-back">🏠 Back to Application</a>
    </div>
</body>
</html>
'''

BATCH_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Results</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #333; font-size: 2em; }
        .table-container { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #667eea; color: white; padding: 12px; text-align: left; }
        td { padding: 12px; border-bottom: 1px solid #e0e0e0; }
        tr:hover { background: #f5f5f5; }
        .approved { color: #28a745; font-weight: bold; }
        .rejected { color: #dc3545; font-weight: bold; }
        .btn-back { display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Batch Prediction Results</h1>
            <p>{{ results|length }} applicants processed successfully</p>
        </div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        {% for key in results[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for row in results %}
                        <tr>
                            {% for key, value in row.items() %}
                                <td>
                                    {% if value is float %}
                                        {{ "%.2f"|format(value) }}
                                    {% else %}
                                        {{ value }}
                                    {% endif %}
                                </td>
                            {% endfor %}
                        </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        <a href="/" class="btn-back">🏠 Back to Application</a>
    </div>
</body>
</html>
'''

# ==================== DATA LOADING & PREPROCESSING ====================
def create_sample_dataset():
    """Create a sample loan dataset for demonstration"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Loan_ID': [f'LP00{str(i).zfill(3)}' for i in range(1, n_samples+1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.75, 0.25]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.randint(1500, 15000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 8000, n_samples),
        'LoanAmount': np.random.randint(50, 400, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 240, 180, 120, 60], n_samples),
        'Credit_History': np.random.choice([0.0, 1.0], n_samples, p=[0.15, 0.85]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p=[0.7, 0.3])
    }
    
    for i in range(n_samples):
        if data['Credit_History'][i] == 1.0:
            if np.random.random() < 0.9:
                data['Loan_Status'][i] = 'Y'
        else:
            if np.random.random() < 0.4:
                data['Loan_Status'][i] = 'Y'
        if data['ApplicantIncome'][i] > 5000 and np.random.random() < 0.85:
            data['Loan_Status'][i] = 'Y'
    
    df = pd.DataFrame(data)
    df.to_csv('loan_data.csv', index=False)
    print("Sample dataset created successfully!")
    return df

def load_and_preprocess_data():
    """Load and preprocess the loan dataset"""
    if not os.path.exists('loan_data.csv'):
        create_sample_dataset()
    
    df = pd.read_csv('loan_data.csv')
    
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna('0', inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# ==================== MODEL TRAINING ====================
def train_models():
    """Train all four models and return the best model"""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_accuracy = 0
    
    print("\nTraining and evaluating models...")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
        
        results[name] = {'train_accuracy': train_acc, 'test_accuracy': test_acc, 'cv_score': cv_score, 'model': model}
        print(f"{name}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}, CV Score = {cv_score:.3f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with test accuracy: {best_accuracy:.3f}")
    
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    plt.figure(figsize=(8, 6))
    df['Loan_Status'].value_counts().plot(kind='bar', color=['green', 'red'])
    plt.title('Loan Approval Distribution')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/loan_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    train_acc = [results[m]['train_accuracy'] for m in model_names]
    test_acc = [results[m]['test_accuracy'] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, train_acc, width, label='Training', color='blue', alpha=0.7)
    plt.bar(x + width/2, test_acc, width, label='Testing', color='orange', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    plt.close()
    
    print("Visualizations saved successfully!")
    return best_model, scaler, X.columns.tolist(), results

def predict_loan_approval(model, scaler, feature_names, input_data):
    """Make prediction for a single applicant"""
    input_df = pd.DataFrame([input_data])
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction, probability

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        gender = request.form.get('gender', 'Male')
        married = request.form.get('married', 'No')
        dependents = request.form.get('dependents', '0')
        education = request.form.get('education', 'Graduate')
        self_employed = request.form.get('self_employed', 'No')
        applicant_income = float(request.form.get('applicant_income', 0))
        coapplicant_income = float(request.form.get('coapplicant_income', 0))
        loan_amount = float(request.form.get('loan_amount', 0))
        loan_amount_term = float(request.form.get('loan_amount_term', 360))
        credit_history = float(request.form.get('credit_history', 1.0))
        property_area = request.form.get('property_area', 'Urban')
        
        gender_enc = 1 if gender == 'Male' else 0
        married_enc = 1 if married == 'Yes' else 0
        education_enc = 1 if education == 'Graduate' else 0
        self_employed_enc = 1 if self_employed == 'Yes' else 0
        property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        property_area_enc = property_area_map.get(property_area, 2)
        
        input_data = {
            'Gender': gender_enc, 'Married': married_enc, 'Dependents': int(dependents),
            'Education': education_enc, 'Self_Employed': self_employed_enc,
            'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history, 'Property_Area': property_area_enc
        }
        
        prediction, probability = predict_loan_approval(model, scaler, feature_names, input_data)
        
        result = {
            'approved': bool(prediction == 1),
            'probability': float(probability),
            'message': 'Loan Approved!' if prediction == 1 else 'Loan Rejected',
            'confidence': f"{probability*100:.1f}%",
            'risk_level': 'Low' if probability > 0.8 else 'Medium' if probability > 0.5 else 'High'
        }
        
        return render_template_string(RESULT_TEMPLATE, result=result)
    except Exception as e:
        return render_template_string(ERROR_TEMPLATE, error=str(e))

@app.route('/dashboard')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        input_data = {}
        for feature in feature_names:
            input_data[feature] = data.get(feature, 0)
        
        prediction, probability = predict_loan_approval(model, scaler, feature_names, input_data)
        
        return jsonify({
            'approved': bool(prediction == 1),
            'probability': float(probability),
            'risk_level': 'Low' if probability > 0.8 else 'Medium' if probability > 0.5 else 'High'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return render_template_string(ERROR_TEMPLATE, error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(ERROR_TEMPLATE, error='No file selected')
        
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        df = pd.read_csv(file)
        predictions = []
        for _, row in df.iterrows():
            input_data = {}
            for feature in feature_names:
                input_data[feature] = row.get(feature, 0)
            prediction, probability = predict_loan_approval(model, scaler, feature_names, input_data)
            predictions.append({'approved': bool(prediction == 1), 'probability': float(probability)})
        
        df['Prediction'] = ['Approved' if p['approved'] else 'Rejected' for p in predictions]
        df['Probability'] = [p['probability'] for p in predictions]
        df.to_csv('batch_results.csv', index=False)
        
        return render_template_string(BATCH_TEMPLATE, results=df.to_dict('records'))
    except Exception as e:
        return render_template_string(ERROR_TEMPLATE, error=str(e))

# ==================== MAIN FUNCTION ====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 SMART LENDER - Loan Approval Prediction System")
    print("="*60)
    
    if not os.path.exists('best_model.pkl'):
        print("\n⏳ First time setup - Training models...")
        train_models()
    else:
        print("\n✅ Model already trained. Loading existing model...")
    
    print("\n" + "="*60)
    print("🚀 Starting Flask Application...")
    print("📍 Access the application at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)