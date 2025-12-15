from typing import Annotated
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, HttpUrl
import model

class Request(BaseModel):
    url: HttpUrl

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset-small/master/test/pants/4aabd82c-82e1-4181-a84d-d0c6e550d26d.jpg"
                }
            ]
        }
    }

class Response(BaseModel):
    prediction: str
    logits: dict[str, float]

app = FastAPI(title="hair-classifier")

@app.post("/predict")
def predict(request: Request) -> Response:
    logits = model.predict(str(request.url))
    prediction = max(logits, key=logits.get)

    return Response(
        prediction=prediction,
        logits=logits,
    )

@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)