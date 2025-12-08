ECR_URL=558947092604.dkr.ecr.eu-north-1.amazonaws.com
LOCAL_IMAGE=churn-prediction-lambda
REPO_URL=${ECR_URL}/${LOCAL_IMAGE}
REGION=eu-north-1

docker build -t ${LOCAL_IMAGE} .

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_URL}

docker tag ${LOCAL_IMAGE}:latest ${REPO_URL}:latest

docker push ${REPO_URL}:latest
