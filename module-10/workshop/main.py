from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import model

class Request(BaseModel):
    url: str = Field(..., example="https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset-small/master/test/pants/4aabd82c-82e1-4181-a84d-d0c6e550d26d.jpg")

app = FastAPI(title="hair-classifier")

@app.post("/predict")
def predict(request: Request) -> dict:
    return model.predict(request.url)

@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)