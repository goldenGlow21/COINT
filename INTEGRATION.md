# Frontend-Backend Integration Guide

## Overview

프론트엔드(React)와 백엔드(Django) 통합이 완료되었습니다.

## Architecture

```
Frontend (React on :3000)
    ↓ HTTP Requests
Backend (Django on :8000)
    ↓ Pipeline
Modules (collector → preprocessor → analyzer → detector)
    ↓ Results
Database (SQLite)
```

## API Endpoints

### Frontend-Compatible Endpoints

#### 1. Get Token List (Home Page)
```
GET /api/tokens/
```

**Response:**
```json
[
  {
    "id": 1,
    "tokenName": "BitDao",
    "Symbol": "BDO",
    "address": "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
    "Type": "Honeypot, Exit",
    "riskScore": 92.5
  }
]
```

#### 2. Get Token Detail (Detail Page)
```
GET /api/tokens/{address}/
```

**Response:**
```json
{
  "id": 1,
  "address": "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
  "tokenName": "BitDao",
  "symbol": "BDO",
  "tokenType": "ERC-20",
  "contractOwner": "0x...",
  "pair": "0x...",
  "riskScore": 92.5,
  "holders": [
    {
      "rank": 1,
      "address": "0x...",
      "percentage": 85
    }
  ],
  "scamTypes": [
    {
      "type": "Honeypot",
      "level": "Warning"
    },
    {
      "type": "Exit",
      "level": "Critical"
    }
  ],
  "victimInsights": [
    {
      "title": "코드 분석: Blacklist",
      "description": "일반 사용자는 매도 불가한 blacklist 로직이 존재함"
    }
  ]
}
```

#### 3. Analyze New Token
```
POST /api/analyze/
Content-Type: application/json

{
  "address": "0x1533C795EA2B33999Dd6eff0256640dC3b2415C2",
  "name": "BitDao"  // optional
}
```

**Response:**
```json
{
  "job_id": 123,
  "status": "pending",
  "message": "Analysis started",
  "address": "0x1533c795ea2b33999dd6eff0256640dc3b2415c2"
}
```

#### 4. Check Analysis Status
```
GET /api/status/{job_id}/
```

**Response:**
```json
{
  "job_id": 123,
  "status": "analyzing",
  "progress": 60,
  "error": null,
  "address": "0x..."
}
```

**Status values:**
- `pending` (0%)
- `collecting` (20%)
- `preprocessing` (40%)
- `analyzing` (60%)
- `detecting` (80%)
- `completed` (100%)
- `failed` (0%)

## Data Flow

### Home Page Data Flow

```
1. Frontend: GET /api/tokens/
2. Backend: Query AnalysisResult where job.status='completed'
3. Backend: Transform to frontend format via TokenListSerializer
4. Frontend: Display token cards with pagination
```

### Detail Page Data Flow

```
1. User searches contract address
2. Frontend: Navigate to /detail?address=0x...
3. Frontend: GET /api/tokens/{address}/
4. Backend: Query AnalysisResult by contract_address
5. Backend: Transform to frontend format via TokenDetailSerializer
6. Frontend: Display detail components (RiskScore, TokenInfo, Holders, etc.)
```

### Analysis Request Flow

```
1. Frontend: POST /api/analyze/ with address
2. Backend: Create AnalysisJob
3. Backend: Trigger pipeline (TODO: async with Celery)
4. Pipeline: collector → preprocessor → analyzer → detector
5. Backend: Save AnalysisResult and DetectedIssues
6. Backend: Update job.status to 'completed'
7. Frontend: Poll /api/status/{job_id}/ until completed
8. Frontend: Navigate to /detail?address=...
```

## Data Mapping

### Backend → Frontend Mapping

**AnalysisResult fields → Frontend fields:**

| Backend | Frontend | Source |
|---------|----------|--------|
| `job.id` | `id` | Direct |
| `job.contract_address` | `address` | Direct |
| `job.contract_name` or `processed_data.token_name` | `tokenName` | Fallback logic |
| `processed_data.symbol` | `Symbol` / `symbol` | JSON field |
| `processed_data.token_type` | `tokenType` | JSON field |
| `raw_data.owner` | `contractOwner` | JSON field |
| `processed_data.pair_address` | `pair` | JSON field |
| `risk_score` | `riskScore` | Direct |
| `analysis_data.holders` | `holders` | JSON array |
| `detection_data.detected_patterns` | `scamTypes` | Transform + severity mapping |
| `DetectedIssue.objects` | `victimInsights` | Related query |

**Severity Mapping:**
```python
{
    'critical': 'Critical',
    'high': 'Critical',
    'medium': 'Warning',
    'low': 'Info',
    'info': 'Info'
}
```

## Development Setup

### 1. Start Backend

```bash
# Activate virtual environment
source venv/bin/activate

# Run migrations (if not done)
python manage.py migrate

# Start Django server
python manage.py runserver
```

Backend will run on `http://localhost:8000`

### 2. Start Frontend

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start React dev server
npm start
```

Frontend will run on `http://localhost:3000`

The proxy is configured to forward API requests to `http://localhost:8000`

## Environment Variables

### Backend (.env)
```env
SECRET_KEY=your-secret-key
DEBUG=True
```

### Frontend (.env.development)
```env
REACT_APP_API_URL=http://localhost:8000/api
```

### Frontend (.env.production)
```env
REACT_APP_API_URL=/api
```

## CORS Configuration

CORS is configured to allow requests from:
- `http://localhost:3000` (React dev server)
- `http://127.0.0.1:3000`

In development mode (`DEBUG=True`), all origins are allowed.

## Module Integration Requirements

### Expected Data Structures

Your analysis modules should populate these fields:

**1. Raw Data (collector output):**
```python
{
    'contract_address': '0x...',
    'source_code': '...',
    'bytecode': '...',
    'abi': [...],
    'owner': '0x...',
    # ... other raw data
}
```

**2. Processed Data (preprocessor output):**
```python
{
    'token_name': 'BitDao',
    'symbol': 'BDO',
    'token_type': 'ERC-20',
    'pair_address': '0x...',
    # ... other processed data
}
```

**3. Analysis Data (analyzer output):**
```python
{
    'holders': [
        {'rank': 1, 'address': '0x...', 'percentage': 85.0},
        {'rank': 2, 'address': '0x...', 'percentage': 8.0},
        # ...
    ],
    'execution_traces': [...],
    'state_changes': [...],
    # ... other analysis results
}
```

**4. Detection Data (detector output):**
```python
{
    'risk_score': 92.5,
    'threat_level': 'high',  # or 'safe', 'low', 'medium', 'critical'
    'detected_patterns': [
        {
            'name': 'Honeypot',
            'severity': 'medium',
            'type': 'Honeypot'
        },
        {
            'name': 'Exit',
            'severity': 'critical',
            'type': 'Exit'
        }
    ],
    'issues': [
        {
            'title': '코드 분석: Blacklist',
            'description': '일반 사용자는 매도 불가한 blacklist 로직이 존재함',
            'severity': 'high',
            'category': 'blacklist',
            'confidence': 0.95
        }
    ]
}
```

## Testing the Integration

### 1. Test with Mock Data

Backend serializers work with the database models. To test:

```bash
# Create a test job via Django admin or shell
python manage.py shell

from api.models import AnalysisJob, AnalysisResult

job = AnalysisJob.objects.create(
    contract_address='0x1533C795EA2B33999Dd6eff0256640dC3b2415C2',
    contract_name='BitDao',
    status='completed'
)

result = AnalysisResult.objects.create(
    job=job,
    raw_data={'owner': '0x...'},
    processed_data={
        'token_name': 'BitDao',
        'symbol': 'BDO',
        'token_type': 'ERC-20',
        'pair_address': '0x...'
    },
    analysis_data={
        'holders': [
            {'rank': 1, 'address': '0x...', 'percentage': 85}
        ]
    },
    detection_data={
        'risk_score': 92.5,
        'detected_patterns': [
            {'name': 'Honeypot', 'severity': 'medium'}
        ],
        'issues': [
            {
                'title': 'Blacklist detected',
                'description': 'Contract has blacklist',
                'severity': 'high'
            }
        ]
    },
    risk_score=92.5,
    threat_level='high'
)
```

### 2. Test API Endpoints

```bash
# Get token list
curl http://localhost:8000/api/tokens/

# Get token detail
curl http://localhost:8000/api/tokens/0x1533C795EA2B33999Dd6eff0256640dC3b2415C2/

# Start analysis
curl -X POST http://localhost:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"address": "0x1234..."}'
```

## TODO

- [ ] Replace mockData imports in frontend with API calls
- [ ] Implement async pipeline execution with Celery
- [ ] Add polling logic in frontend for analysis status
- [ ] Add loading states and error handling in frontend
- [ ] Set up production deployment (Nginx, Gunicorn, PostgreSQL)
- [ ] Add authentication if needed
- [ ] Implement rate limiting
- [ ] Add comprehensive error messages
