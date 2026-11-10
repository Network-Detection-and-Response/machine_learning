FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY ndr_ml/ ./ndr_ml/
RUN pip install --no-cache-dir .
COPY models/ ./models/
EXPOSE 8000
CMD ["uvicorn", "ndr_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
