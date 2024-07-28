"""
Datos de entrada del modelo:
['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'gender_Other', 'ever_married_Yes',
       'work_type_Never_worked', 'work_type_Private',
       'work_type_Self-employed', 'work_type_children', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes']

{
    'age': int,
    'hypertension': int (1/0),
    'gender': str (male/female/other),
    'ever_married_Yes': int (1/0),
    'heart_disease': int (1/0),
    'avg_glucose_level': int,
    'bmi': int,
    'work_type': str (never worked/private/self-employed/children)
    'residence_type': str (urban)
    'smoking_status': str (formerly smoked/never smoked/smokes)
}

{
    "age": 33,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 0,
    "avg_glucose_level": 70,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

{
    "age": 75,
    "hypertension": 1,
    "gender": "male",
    "ever_married_Yes": 1,
    "heart_disease": 1,
    "avg_glucose_level": 120,
    "bmi": 29,
    "work_type": "private",
    "residence_type": "urban",
    "smoking_status": "never smoked"
}

"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import jwt

SECRET_KEY = "MLOPS"  
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


model = joblib.load('model.sav')


def verify_token(token: str):
    if token != "mysecrettoken":
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post('/token')
def login():
    return {"access_token": "mysecrettoken", "token_type": "bearer"}


class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Age must be between 18 and 120")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension must be 0 or 1")
    gender: str = Field(..., pattern="^(male|female|other)$", description="Gender must be male, female, or other")
    ever_married_Yes: int = Field(..., ge=0, le=1, description="Ever married must be 0 or 1")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease must be 0 or 1")
    avg_glucose_level: float = Field(..., gt=0, description="Average glucose level must be greater than 0")
    bmi: float = Field(..., gt=0, description="BMI must be greater than 0")
    work_type: str = Field(..., pattern="^(never worked|private|self-employed|children)$", description="Work type must be one of 'never worked', 'private', 'self-employed', 'children'")
    residence_type: str = Field(..., pattern="^(urban|rural)$", description="Residence type must be urban or rural")
    smoking_status: str = Field(..., pattern="^(never smoked|occasionally smoked|smokes)$", description="Smoking status must be 'never smoked', 'occasionally smoked', or 'smokes'")




def gender_encoding(message):
    gender_encoded = {'gender_Male': 0, 'gender_Other': 0}
    if message['gender'].lower() == 'male':
        gender_encoded['gender_Male'] = 1
    elif message['gender'].lower() == 'other':
        gender_encoded['gender_Other'] = 1

    del message['gender']
    message.update(gender_encoded)
    return message

def work_type_encoding(message):
    work_type_encoded = {'work_type_Never_worked': 0, 'work_type_Private': 0,
                         'work_type_Self-employed': 0, 'work_type_children': 0}

    if message['work_type'].lower() == 'never worked':
        work_type_encoded['work_type_Never_worked'] = 1
    elif message['work_type'].lower() == 'private':
        work_type_encoded['work_type_Private'] = 1
    elif message['work_type'].lower() == 'self-employed':
        work_type_encoded['work_type_Self-employed'] = 1
    elif message['work_type'].lower() == 'children':
        work_type_encoded['work_type_children'] = 1

    del message['work_type']
    message.update(work_type_encoded)
    return message

def residence_encoding(message):
    residence_encoded = {'Residence_type_Urban': 0}
    if message['residence_type'] == 'urban':
        residence_encoded['Residence_type_Urban'] = 1

    del message['residence_type']
    message.update(residence_encoded)
    return message

def smoking_encoding(message):
    smoking_encoded = {'smoking_status_formerly smoked': 0, 'smoking_status_never smoked': 0,
                       'smoking_status_smokes': 0}
    if message['smoking_status'] == 'formerly smoked':
        smoking_encoded['smoking_status_formerly smoked'] = 1
    elif message['smoking_status'] == 'never smoked':
        smoking_encoded['smoking_status_never smoked'] = 1
    elif message['smoking_status'] == 'smokes':
        smoking_encoded['smoking_status_smokes'] = 1

    del message['smoking_status']
    message.update(smoking_encoded)
    return message

def data_prep(message):
    message = gender_encoding(message)
    message = work_type_encoding(message)
    message = residence_encoding(message)
    message = smoking_encoding(message)
    return pd.DataFrame(message, index=[0])

def heart_prediction(message: dict):
    data = data_prep(message)
    label = model.predict(data)[0]
    return {'label': int(label)}

@app.get('/')
def main():
    return {'message': 'Hola'}

@app.post('/heart-attack-prediction/')
def predict_heart_attack(message: PatientData, token: str = Depends(oauth2_scheme)):
    verify_token(token)
    model_pred = heart_prediction(message.dict())
    return model_pred