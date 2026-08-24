FROM python:3.12-slim

WORKDIR /opt/project

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.integration:app", "--host", "0.0.0.0", "--port", "8000"]