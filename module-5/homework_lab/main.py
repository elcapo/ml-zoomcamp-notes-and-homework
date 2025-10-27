import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

with open("pipeline_v1.bin", "rb") as f:
    dict_vectorizer, model = pickle.load(f)

app = FastAPI()

@app.post("/predict")
def predict(client: Client) -> float:
    x = dict_vectorizer.transform(client.model_dump())
    y = model.predict_proba(x)[:, 1]

    return y[0]