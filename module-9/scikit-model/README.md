# Scikit-Learn Model

## Naked Python Environments

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start

```bash
python -m model_package.api
```

### Use

```bash
# Generate random data for a prediction
random_sample=`curl -X GET http://127.0.0.1:5000/generate`

# Make a prediction
curl -X POST -H "Content-Type: application/json" -d $random_sample http://127.0.0.1:5000/predict
```

## Local Docker Environment

### Install

```bash
docker build -t churn-prediction-lambda .
```

### Start

```bash
docker run -it --rm -p 8080:8080  churn-prediction-lambda
```

### Use

```bash
# Make a prediction
curl -X POST -H "Content-Type: application/json" -d $random_sample http://localhost:8081/2015-03-31/functions/function/invocations
```
