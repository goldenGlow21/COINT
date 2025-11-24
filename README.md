# COINT - Token Scam Detection System

Ethereum token scam detection platform.

## Setup

```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Run server
python manage.py runserver
```

## Environment Variables

Create `.env` file:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
```

## Project Structure

```
api/            Django app (models, views, serializers)
pipeline/       Analysis pipeline (adapters, orchestrator)
modules/        Analysis modules (to be integrated)
config/         Django settings
frontend/       React frontend (separate)
```
