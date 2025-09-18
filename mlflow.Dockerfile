# mlflow.Dockerfile
FROM python:3.10.12-bookworm

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow

EXPOSE 5001