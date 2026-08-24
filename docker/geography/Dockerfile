FROM python:3.12-slim

WORKDIR /opt/project

RUN pip install --no-cache-dir \
    "setuptools<81" \
    "mlflow==2.14.1" \
    fastapi \
    "uvicorn[standard]" \
    pandas \
    pyarrow \
    scikit-learn

ENV PYTHONUNBUFFERED=1

COPY . .

CMD ["uvicorn", "api.routers.geography:app", "--host", "0.0.0.0", "--port", "8000"]
