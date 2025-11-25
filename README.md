# Sentiment Analysis API

Dockerized Flask API for sentiment analysis using HuggingFace transformers, featuring Redis caching and comprehensive health checks.

## Quick Start

### Option A: Docker (Recommended)

1.  **Clone & Setup**:
    ```bash
    # Clone repository
    git clone https://github.com/LAG-4/carnot-assignment.git
    cd carnot-assignment
    
    # Copy environment variables
    cp .env.example .env       # Linux/Mac/PowerShell
    # copy .env.example .env   # Windows CMD
    ```

2.  **Run**:
    ```bash
    docker-compose up --build
    ```
    The API will be available at `http://localhost:5000`.

### Option B: Local Development

**Prerequisites**: Python 3.12+, `uv` (optional but recommended), Redis.

1.  **Setup Environment**:
    ```bash
    # Make sure uv is installed

    # Sync dependencies & activate
    uv sync

    # Copy environment variables
    cp .env.example .env       # Linux/Mac/PowerShell
    # copy .env.example .env   # Windows CMD

    # Activate virtual environment
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2.  **Run Redis**:
    ```bash
    # Run Redis via Docker
    docker run -d -p 6379:6379 redis:7-alpine
    ```

3.  **Run App**:
    ```bash
    python app.py
    # Or with Gunicorn:
    # gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 app:app
    ```

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API Info |
| `GET` | `/health` | Health Check |
| `POST` | `/predict` | Analyze single text |
| `POST` | `/batch-predict` | Analyze multiple texts |

**Example Request:**
You can use postman/thunder client or curl to check the api results.
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "use_cache": true}'
```
```
#for batch processing
curl -X POST http://localhost:5000/batch-predict   -H "Content-Type: application/json"   -d '{
    "texts": [
      "I love this product",
      "I hate it. This is not for me",
      "Would recommend it to my friends"
    ],
    "use_cache": true
  }'
```
## Features & Architecture

- **Stack**: Flask, Gunicorn, DistilBERT (SST-2), Redis.
- **Key Features**: 
    - **Sentiment Analysis**: DistilBERT model for accurate classification.
    - **Caching**: Redis caching for repeated queries.
    - **Production Ready**: Health checks, structured logging, non-root Docker user.
- **Configuration**: Managed via `.env` file (see `.env.example`).

## Management

- **Logs**: `docker-compose logs -f api`
- **Stop**: `docker-compose down`
- **Test**: `curl http://localhost:5000/health`
