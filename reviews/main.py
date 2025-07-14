from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import uvicorn

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / 'model_reviews.pkl'
vec_path = BASE_DIR / 'vec_reviews.pkl'

model = joblib.load(model_path)
vec = joblib.load(vec_path)


app = FastAPI(title='Bank Reviews NLP')

class Bank_reviews(BaseModel):
    text: str

@app.post('/reviews/')
async def bank_reviews(text: Bank_reviews):
    features = list(text.dict().values())
    vec_data = vec.transform(features)
    pred = model.predict(vec_data)[0]

    return{'answer': pred}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)