# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./

# Install poetry (dependency manager)
RUN pip install --no-cache-dir poetry
RUN poetry self sync
RUN poetry install --only main

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]