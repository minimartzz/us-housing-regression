# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./

# Install poetry (dependency manager)
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry install --sync --only main

# Copy project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run API with Uvicorn
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]