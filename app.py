import base64
import difflib
from io import BytesIO

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Configure matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load models and data
def load_models():
    model = joblib.load('assets/optimized_disease_predictor.pkl')
    label_encoder = joblib.load('assets/label_encoder.pkl')
    selected_features = joblib.load('assets/selected_features.pkl')
    return model, label_encoder, selected_features

model, label_encoder, selected_features = load_models()

# Use selected features as the complete list of symptoms
ALL_SYMPTOMS = selected_features

# Pydantic model for input validation
class SymptomsInput(BaseModel):
    symptoms: list[str]

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI Disease Predictor"}

@app.post("/predict")
def predict_disease(data: SymptomsInput):
    try:
        if not data.symptoms:
            return JSONResponse(content={"error": "Please provide a list of symptoms"}, status_code=400)

        user_symptoms = {}
        for symptom in data.symptoms:
            matched = get_closest_symptom(symptom)
            if matched:
                user_symptoms[matched] = 1

        if not user_symptoms:
            return JSONResponse(content={"error": "No valid symptoms provided"}, status_code=400)

        prediction = get_prediction(user_symptoms)
        graph_image = create_prediction_graph(prediction)
        symptoms_chart = create_symptoms_chart(user_symptoms)

        return {
            "symptoms_reported": list(user_symptoms.keys()),
            "prediction": prediction,
            "graph_image_base64": graph_image,
            "symptoms_chart_base64": symptoms_chart
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Utility functions
def get_closest_symptom(symptom: str):
    symptom = symptom.replace(' ', '_').lower()
    matches = difflib.get_close_matches(symptom, ALL_SYMPTOMS, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_prediction(symptoms: dict):
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    probabilities = model.predict_proba(input_data)[0]
    top3_indices = probabilities.argsort()[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]

    return {
        'primary_prediction': {
            'disease': top3_diseases[0],
            'confidence': float(top3_probs[0])
        },
        'alternative_predictions': [
            {'disease': disease, 'confidence': float(prob)}
            for disease, prob in zip(top3_diseases[1:], top3_probs[1:])
        ]
    }

def create_prediction_graph(prediction):
    diseases = [prediction['primary_prediction']['disease']]
    confidences = [prediction['primary_prediction']['confidence']]
    for alt in prediction['alternative_predictions']:
        diseases.append(alt['disease'])
        confidences.append(alt['confidence'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=diseases, y=confidences, palette="viridis")
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{confidences[i]:.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.title('Prediction Confidence', fontsize=16)
    plt.xlabel('')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()

    return save_plot_to_base64()

def create_symptoms_chart(symptoms):
    symptom_list = list(symptoms.keys())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[1]*len(symptom_list), y=symptom_list, orient='h', palette="rocket")
    plt.title('Reported Symptoms', fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.tight_layout()

    return save_plot_to_base64()

def save_plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
