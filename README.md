# COINT - Contract Intelligence Platform

Ethereum smart contract analysis platform with data collection, preprocessing, dynamic analysis, and rule-based threat detection.

## Project Structure

```
COINT/
├── api/                    # Django REST API
│   ├── models.py          # Database models for jobs, results, issues
│   ├── serializers.py     # API data serializers
│   ├── views.py           # API endpoints
│   └── urls.py            # URL routing
│
├── modules/               # Analysis modules
│   ├── collector/         # Ethereum data collection
│   ├── preprocessor/      # Data preprocessing
│   ├── analyzer/          # Dynamic code analysis
│   └── detector/          # Rule-based threat detection
│
├── pipeline/              # Pipeline orchestration
│   ├── interfaces.py      # Base interfaces for modules
│   ├── adapters.py        # Module adapters
│   └── orchestrator.py    # Pipeline coordinator
│
├── config/                # Django configuration
├── data/                  # Data storage (raw, processed, results)
└── logs/                  # Application logs
```

## Technology Stack

- **Backend**: Django 5.2.7 + Django REST Framework 3.16.1
- **Database**: SQLite (upgradeable to PostgreSQL)
- **Language**: Python 3.13
- **Frontend**: React (separate repository)

## Setup

### 1. Environment Setup

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create `.env` file in project root:

```env
SECRET_KEY=your-secret-key-here
DEBUG=True
```

### 3. Database Setup

```bash
# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### 4. Run Development Server

```bash
python manage.py runserver
```

API will be available at `http://localhost:8000/api/`

## API Endpoints

### Analysis Jobs

- `POST /api/jobs/` - Create new analysis job
  ```json
  {
    "contract_address": "0x...",
    "contract_name": "MyContract"
  }
  ```

- `GET /api/jobs/` - List all jobs
- `GET /api/jobs/{id}/` - Get job details
- `GET /api/jobs/{id}/status/` - Get job status
- `GET /api/jobs/{id}/result/` - Get analysis result
- `DELETE /api/jobs/{id}/` - Delete job

### Results

- `GET /api/results/` - List all results
- `GET /api/results/{id}/` - Get specific result

### Issues

- `GET /api/issues/` - List all detected issues
- `GET /api/issues/{id}/` - Get specific issue
- `GET /api/issues/?result={id}` - Filter issues by result

## Module Integration

### Adding Analysis Modules

1. Place your module code in the corresponding directory:
   - `modules/collector/` - Data collection
   - `modules/preprocessor/` - Preprocessing
   - `modules/analyzer/` - Dynamic analysis
   - `modules/detector/` - Threat detection

2. Update the adapter in `pipeline/adapters.py`:

```python
# Example: Integrating collector module
from modules.collector import YourCollector

class CollectorAdapter(DataCollector):
    def __init__(self):
        self.collector = YourCollector()

    def collect(self, contract_address: str) -> Dict[str, Any]:
        return self.collector.fetch_data(contract_address)
```

3. Ensure your module output matches the expected data structure (see interfaces.py)

## Data Flow

```
1. Data Collection    → raw_data
2. Preprocessing      → processed_data
3. Dynamic Analysis   → analysis_data
4. Threat Detection   → detection_data + risk_score
5. Database Storage   → AnalysisResult + DetectedIssue
```

## Development

### Running Tests

```bash
python manage.py test
```

### Admin Interface

Access Django admin at `http://localhost:8000/admin/`

View and manage:
- Analysis jobs
- Results
- Detected issues

## Database Models

### AnalysisJob
Tracks analysis job status and metadata

### AnalysisResult
Stores complete analysis data from all pipeline stages

### DetectedIssue
Individual security issues found during analysis

## Logging

Logs are stored in `logs/pipeline.log`

Configure logging level in `config/settings.py`:
```python
LOGGING = {
    'loggers': {
        'pipeline': {
            'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
        },
    },
}
```

## TODO

- [ ] Integrate actual analysis modules into `modules/` directories
- [ ] Update adapters in `pipeline/adapters.py`
- [ ] Implement async job execution (Celery)
- [ ] Add CORS middleware for frontend integration
- [ ] Set up PostgreSQL for production
- [ ] Add authentication and rate limiting
- [ ] Write comprehensive tests

## Notes

- All module implementations currently use placeholder data
- Update adapters as you integrate real modules
- File I/O in modules should be replaced with in-memory data passing
- Consider adding Celery for async pipeline execution
