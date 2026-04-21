import joblib, numpy as np, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app         = FastAPI(title='Titanic Survival Predictor')
model       = joblib.load('models/best_model.pkl')
transformer = joblib.load('models/transformer.pkl')


class PassengerInput(BaseModel):
    pclass:   int                    # 1, 2, or 3
    sex:      str                    # male / female
    age:      Optional[float] = None
    sibsp:    int   = 0
    parch:    int   = 0
    fare:     float = 32.0
    embarked: Optional[str]  = 'S'  # S / C / Q
    title:    Optional[str]  = None # Mr / Mrs / Miss / Master / Officer / Royalty / Sir


@app.get('/')
def root():
    return {'status': 'ok', 'model': 'titanic-survival'}


@app.post('/predict')
def predict(data: PassengerInput):
    age       = data.age if data.age is not None else 29.0
    fare      = data.fare
    family_sz = data.sibsp + data.parch + 1
    is_alone  = int(family_sz == 1)
    fare_log  = float(np.log1p(fare))
    title     = data.title or ('Mr' if data.sex == 'male' else 'Mrs')

    row = pd.DataFrame([{
        'age':         age,
        'fare':        fare,
        'fare_log':    fare_log,
        'sibsp':       data.sibsp,
        'parch':       data.parch,
        'family_size': family_sz,
        'is_alone':    is_alone,
        'pclass':      data.pclass,
        'sex':         data.sex,
        'embarked':    data.embarked,
        'title':       title,
    }])

    X    = transformer.transform(row)
    prob = float(model.predict_proba(X)[0, 1])
    return {
        'prediction':  'survived' if prob >= 0.5 else 'died',
        'probability': round(prob, 4)
    }
