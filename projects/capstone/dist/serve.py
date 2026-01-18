import random
import pandas as pd
from fastapi import FastAPI
from api.ping_response import PingResponse
from api.poverty_risk_request import PovertyRiskRequest
from api.poverty_risk_response import PovertyRiskResponse
from api.model import load_model

app = FastAPI()
model, imputer, scaler = load_model()


@app.get("/ping")
def ping() -> PingResponse:
    """Simple endpoint to check that the API service is alive"""

    return PingResponse(alive=True)


@app.post("/predict")
def predict(request: PovertyRiskRequest) -> PovertyRiskResponse:
    """Predict if a household is in poverty risk"""

    df = pd.DataFrame([request.model_dump()])
    df = imputer.fit_transform(df)
    result = model.predict(df)

    return PovertyRiskResponse(request=request, poverty_risk=result[0] == 1)


@app.get("/random_request")
def random_request() -> PovertyRiskRequest:
    """Generate a random request"""

    return PovertyRiskRequest(
        total_disposable_income=random.uniform(-20000, 600000),
        income_before_all_social_transfers=random.uniform(-15000, 500000),
        head_age=random.randint(16, 90),
        head_sex=random.randint(1, 2),
        head_marital_status=random.randint(1, 5),
        head_education_level=random.choice([10, 20, 30, 40, 50]),
        head_employment_status=random.choice([43, 52, 23, 51, 42, 83, 33, 91]),
        head_hours_worked_per_week=random.randint(2, 84),
        head_net_employee_income=random.uniform(0, 300000),
        head_general_health_status=random.randint(1, 5),
        household_size=random.randint(1, 10),
        mean_age=random.uniform(18, 90),
        std_age=random.uniform(0, 45),
        num_males=random.randint(0, 5),
        max_education=random.choice([50, 45, 40, 35, 30, 20, 10]),
        num_employed=random.randint(0, 2),
        total_employee_income=random.uniform(0, 400000),
        total_pension=random.uniform(0, 150000),
        avg_health=random.randint(1, 5),
        employment_rate=random.randint(0, 1),
        num_children=random.randint(0, 2),
        num_elderly=random.randint(0, 4),
        num_working_age=random.randint(0, 8),
        dependency_ratio=random.randint(0, 4),
        income_per_capita=random.uniform(0, 250000),
    )
