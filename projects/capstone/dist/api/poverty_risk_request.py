from pydantic import BaseModel


class PovertyRiskRequest(BaseModel):
    total_disposable_income: float
    income_before_all_social_transfers: float
    head_age: int
    head_sex: int
    head_marital_status: int
    head_education_level: int
    head_employment_status: int
    head_hours_worked_per_week: int
    head_net_employee_income: float
    head_general_health_status: int
    household_size: int
    mean_age: float
    std_age: float
    num_males: int
    max_education: int
    num_employed: int
    total_employee_income: float
    total_pension: float
    avg_health: float
    employment_rate: int
    num_children: int
    num_elderly: int
    num_working_age: int
    dependency_ratio: int
    income_per_capita: float
