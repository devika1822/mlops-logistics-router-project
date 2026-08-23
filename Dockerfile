FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY api/ /app/api/
COPY tracks/vehicle/ /app/tracks/vehicle/

EXPOSE 8000

CMD ["uvicorn", "api.app_telemetry_da25g503:app", "--host", "0.0.0.0", "--port", "8000"]
