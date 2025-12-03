# COINT - Token Scam Detection System

Ethereum 토큰 스캠 탐지 플랫폼

## System Overview

COINT는 3가지 ML 기반 탐지 모듈을 통합한 토큰 스캠 분석 시스템입니다:

1. **Honeypot Detection (Dynamic Analysis)**: Brownie 기반 스마트 컨트랙트 시뮬레이션 테스트
2. **Honeypot Detection (ML)**: XGBoost v8 모델 (67 features, 96% accuracy)
3. **Exit Scam Detection (ML)**: Attention-based MIL 모델 (거래 패턴 분석)

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
```env
SECRET_KEY=your-django-secret-key
DEBUG=True

# Blockchain data collection
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
ETHERSCAN_API_URL=https://api.etherscan.io/v2/api
```

## Project Structure

```
api/                    Django app (models, views, serializers)
├── models.py          Database schema (11 tables)
├── migrations/        DB migrations (0001~0005)
└── serializers.py     REST API serializers

pipeline/              Analysis pipeline orchestration
├── adapters.py        Module integration adapters
└── orchestrator.py    Pipeline coordinator

modules/               Analysis modules
├── data_collector/    Unified blockchain data collector
├── honeypot_DA/       Dynamic analysis (Brownie-based)
├── honeypot_ML/       ML-based honeypot detection (XGBoost)
├── exit_ML/           Exit scam detection (Attention MIL)
└── preprocessor/      Feature engineering (TBD)

config/                Django settings
frontend/              React frontend (separate repository)
```

## Database Schema

11 테이블로 구성:

### Raw Data (3 tables)
- `token_info`: 토큰 메타데이터 및 페어 정보
- `pair_evt`: 페어 이벤트 로그 (Mint, Burn, Swap, Sync)
- `holder_info`: 토큰 홀더 정보

### Processed Data (2 tables)
- `honeypot_processed_data`: Honeypot 탐지 피처 (23개)
- `exit_processed_data`: Exit scam 탐지 피처 (52개, 5초 윈도우)

### Analysis Results (5 tables)
- `honeypot_da_result`: 동적 분석 결과
- `honeypot_ml_result`: ML 기반 honeypot 탐지 결과
- `exit_ml_result`: Exit scam 탐지 결과
- `exit_ml_detect_transaction`: 거래별 탐지 상세
- `exit_ml_detect_static`: 윈도우별 정적 피처

### Final Output (1 table)
- `result`: 통합 분석 결과 및 리스크 스코어

## Integration Status

### ✅ Completed
- Database schema design and migrations
- UnifiedDataCollector (token/pair/holder data collection)
- HoneypotDAAnalyzerAdapter (8 test scenarios)
- HoneypotMLAnalyzerAdapter (XGBoost v8, threshold 0.64)
- ExitMLAnalyzerAdapter (Attention MIL model)
- Environment variable management (.env, settings.py)

### 🚧 In Progress
- Preprocessor module (feature engineering)
- API endpoints (REST API)
- Pipeline orchestrator

### 📋 Planned
- Frontend integration
- Real-time monitoring
- Result caching and optimization

## Technologies

**Backend:**
- Django 5.2.7 + Django REST Framework
- PostgreSQL (production) / SQLite (development)

**Blockchain:**
- Web3.py 6.20.0 (Ethereum interaction)
- Etherscan API v2 (data collection)
- Brownie (smart contract testing)

**Machine Learning:**
- XGBoost 1.7.6 (honeypot detection)
- PyTorch 2.9.1 (exit scam detection)
- Pandas, NumPy, scikit-learn

**Frontend:**
- React (separate repository)

## API Usage (WIP)

```python
from pipeline.adapters import DataCollectorAdapter

# Collect blockchain data
collector = DataCollectorAdapter()
data = collector.collect_all("0x...")  # token address
token_info = collector.save_to_db(data)

# Run analysis (after preprocessor integration)
# from pipeline.orchestrator import AnalysisPipeline
# pipeline = AnalysisPipeline()
# result = pipeline.analyze(token_addr="0x...")
```

## Development

```bash
# Run tests
python manage.py test

# Create new migration
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

## License

Proprietary - BoB Project
