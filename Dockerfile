FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY api/ /app/api/
COPY tracks/vehicle/ /app/tracks/vehicle/

EXPOSE 8000

CMD ["uvicorn", "api.app_telemetry_da25g503:app", "--host", "0.0.0.0", "--port", "8000"]
