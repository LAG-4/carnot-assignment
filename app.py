import os
import logging
import time
import json
import hashlib
from datetime import datetime
from functools import wraps
import redis
from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 3600))  # 1 hour default
MODEL_NAME = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5
    )
    redis_client.ping()
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

logger.info(f"Loading sentiment analysis model: {MODEL_NAME}")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1  # Use CPU
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sentiment_analyzer = None


def log_request(f):
    """Decorator to log API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

        response = f(*args, **kwargs)

        duration = time.time() - start_time
        logger.info(f"Response: {request.path} completed in {duration:.3f}s")

        return response
    return decorated_function


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns the health status of the API and its dependencies
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'sentiment-analysis-api',
        'version': '1.0.0',
        'checks': {}
    }


    if sentiment_analyzer is not None:
        health_status['checks']['model'] = 'healthy'
    else:
        health_status['checks']['model'] = 'unhealthy'
        health_status['status'] = 'unhealthy'

    if redis_client is not None:
        try:
            redis_client.ping()
            health_status['checks']['redis'] = 'healthy'
        except redis.ConnectionError:
            health_status['checks']['redis'] = 'unhealthy'
            health_status['status'] = 'degraded'
    else:
        health_status['checks']['redis'] = 'unavailable'
        health_status['status'] = 'degraded'

    status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
    return jsonify(health_status), status_code





@app.route('/predict', methods=['POST'])
@log_request
def predict_sentiment():
    """
    Sentiment analysis prediction endpoint

    Request body:
    {
        "text": "Text to analyze",
        "use_cache": true  (optional, default: true)
    }

    Response:
    {
        "text": "Input text",
        "sentiment": "POSITIVE" or "NEGATIVE",
        "confidence": 0.99,
        "cached": false,
        "processing_time": 0.123
    }
    """
    start_time = time.time()


    if sentiment_analyzer is None:
        logger.error("Model not loaded")
        return jsonify({
            'error': 'Model not available',
            'message': 'Sentiment analysis model is not loaded'
        }), 503

    data = request.get_json()

    if not data or 'text' not in data:
        logger.warning("Missing 'text' in request")
        return jsonify({
            'error': 'Bad request',
            'message': "'text' field is required"
        }), 400

    text = data['text']
    use_cache = data.get('use_cache', True)

    if not text or not isinstance(text, str):
        return jsonify({
            'error': 'Bad request',
            'message': "'text' must be a non-empty string"
        }), 400

    if len(text) > 5000:
        return jsonify({
            'error': 'Bad request',
            'message': "'text' exceeds maximum length of 5000 characters"
        }), 400

  
    cached = False
    cache_key = f"sentiment:{hashlib.md5(text.encode('utf-8')).hexdigest()}"

    if use_cache and redis_client is not None:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for text: {text[:50]}...")
                
                result = json.loads(cached_result)
                result['cached'] = True
                result['processing_time'] = time.time() - start_time
                return jsonify(result), 200
        except Exception as e:
            logger.warning(f"Cache read error: {e}")


    try:
        logger.info(f"Analyzing sentiment for text: {text[:50]}...")
        prediction = sentiment_analyzer(text)[0]

        result = {
            'text': text,
            'sentiment': prediction['label'],
            'confidence': round(prediction['score'], 4),
            'cached': cached,
            'processing_time': time.time() - start_time
        }


        if use_cache and redis_client is not None:
            try:
                
                redis_client.setex(
                    cache_key,
                    CACHE_EXPIRY,
                    json.dumps({
                        'text': text,
                        'sentiment': prediction['label'],
                        'confidence': round(prediction['score'], 4)
                    })
                )
                logger.info(f"Cached result for text: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch-predict', methods=['POST'])
@log_request
def batch_predict_sentiment():
    """
    Batch sentiment analysis endpoint

    Request body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "use_cache": true  (optional, default: true)
    }

    Response:
    {
        "results": [
            {"text": "...", "sentiment": "POSITIVE", "confidence": 0.99},
            ...
        ],
        "total": 2,
        "cached_count": 1,
        "processing_time": 0.234
    }
    """
    start_time = time.time()


    if sentiment_analyzer is None:
        return jsonify({
            'error': 'Model not available',
            'message': 'Sentiment analysis model is not loaded'
        }), 503


    data = request.get_json()

    if not data or 'texts' not in data:
        return jsonify({
            'error': 'Bad request',
            'message': "'texts' field is required"
        }), 400

    texts = data['texts']
    use_cache = data.get('use_cache', True)

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({
            'error': 'Bad request',
            'message': "'texts' must be a non-empty list"
        }), 400

    if len(texts) > 100:
        return jsonify({
            'error': 'Bad request',
            'message': "'texts' list exceeds maximum length of 100"
        }), 400

    results = []
    cached_count = 0

    for text in texts:
        if not isinstance(text, str) or not text:
            continue

        # Try cache first
        cache_key = f"sentiment:{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        cached_result = None

        if use_cache and redis_client is not None:
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                   
                    result = json.loads(cached_result)
                    results.append(result)
                    cached_count += 1
                    continue
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        # Analyze if not cached
        try:
            prediction = sentiment_analyzer(text)[0]
            result = {
                'text': text,
                'sentiment': prediction['label'],
                'confidence': round(prediction['score'], 4)
            }
            results.append(result)

            # Cache the result
            if use_cache and redis_client is not None:
                try:
                    
                    redis_client.setex(
                        cache_key,
                        CACHE_EXPIRY,
                        json.dumps(result)
                    )
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

        except Exception as e:
            logger.error(f"Prediction error for text: {e}")
            results.append({
                'text': text,
                'error': str(e)
            })

    return jsonify({
        'results': results,
        'total': len(results),
        'cached_count': cached_count,
        'processing_time': time.time() - start_time
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Dockerized Sentiment Analysis API With Caching and Health Checks',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'batch_predict': '/batch-predict (POST)'
        },
        'model': MODEL_NAME
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting app {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
