from pydantic import BaseModel
from api.poverty_risk_request import PovertyRiskRequest


class PovertyRiskResponse(BaseModel):
    request: PovertyRiskRequest
    poverty_risk: bool
