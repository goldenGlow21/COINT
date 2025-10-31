# System Architecture Documentation

## Overview

This document describes the complete architecture of the token scam detection system, designed according to the provided DB schema and workflow specifications.

## Database Schema

The system uses 7 main tables to store analysis data:

### 1. token_info
Central reference table for all analyzed tokens.
- `id` (token_addr_idx): Auto-generated primary key
- `token_addr`: Token contract address (unique)
- `pair_addr`: Pair contract address
- `token_create_ts`: Token creation timestamp
- `lp_create_ts`: Liquidity pool creation timestamp
- `pair_idx`: Token index in pair (0 or 1)
- `pair_type`: Router type (e.g., UniswapV2)

### 2. analysis_job
Tracks pipeline execution status for each analysis request.
- `id`: Job identifier
- `token_addr`: Token being analyzed
- `token_info_id`: Foreign key to token_info (after collection)
- `status`: Current pipeline stage
- `current_step`: Human-readable description
- `created_at`, `updated_at`, `completed_at`: Timestamps
- `error_message`, `error_step`: Error tracking

**Status flow:**
pending → collecting_token → collecting_pair → collecting_holder →
preprocessing → analyzing_honeypot_da → analyzing_honeypot_ml →
analyzing_exit_ml → aggregating → completed/failed

### 3. result
Final analysis results returned to frontend.
- `id`: Result identifier
- `token_addr`: Analyzed token address (unique)
- `token_info_id`: One-to-one with token_info
- `risk_score`: Float (0-100)
- `scam_types`: JSON array of detected scam categories
- `victim_insights`: JSON array of detailed findings
- `created_at`: Analysis completion timestamp

### 4. pair_evt
Raw blockchain event data for the token's liquidity pair.
- `token_info_id`: Foreign key (multiple events per token)
- `timestamp`: Event timestamp
- `block_number`: Block number
- `tx_hash`, `tx_from`, `tx_to`: Transaction details
- `evt_idx`: Event log index
- `evt_type`: Event type (Mint, Burn, Sync, Swap)
- `evt_log`: JSON of preprocessed event data
- `token0`, `token1`: Pair token addresses
- `reserve0`, `reserve1`: Current reserves
- `lp_total_supply`: Current LP token supply

### 5. holder_info
Token holder distribution data.
- `token_info_id`: Foreign key (multiple holders per token)
- `holder_addr`: Wallet address
- `balance`: Token balance (Decimal)
- `rel_to_total`: Percentage string (e.g., "15.3%")

### 6. honeypot_processed_data
Preprocessed features for honeypot detection (23 features).
- One record per token (one-to-one with token_info)
- Trade counts, imbalance metrics, event counts
- Volume metrics (including log-transformed)
- Additional metrics (liquidity events, seller/buyer counts, etc.)

### 7. exit_processed_data
Preprocessed features for exit scam detection (52 features per window).
- Multiple records per token (one per 5-second window)
- Window metadata (win_id, win_start_ts, win_start_block, etc.)
- 5-second window metrics (LP, reserve, events, swaps)
- 60-second rolling window metrics
- 600-second rolling window metrics
- Holder concentration metrics

## Pipeline Architecture

### Workflow Execution

```
1. User submits token_addr
2. Check result table for existing analysis
   ├─ If exists: Return cached result immediately
   └─ If not exists: Start pipeline
3. Collect token metadata → TokenInfo
4. Collect pair events → PairEvent records
5. Collect holder information → HolderInfo records
6. Preprocess data → HoneypotProcessedData + ExitProcessedData
7. Run honeypot_DA → Analysis results (in-memory)
8. Run honeypot_ML → Prediction results (in-memory)
9. Run exit_ML → Prediction results (in-memory)
10. Aggregate results → Result table
11. Return result to frontend
```

### Module Integration Points

#### 1. collector_token
- Location: `modules/collector_token/`
- Input: `token_addr` (string)
- Output: Dictionary with token metadata
- Database: Creates TokenInfo record
- Adapter: `TokenCollectorAdapter`

#### 2. collector_pair
- Location: `modules/collector_pair/`
- Input: TokenInfo instance
- Output: List of event dictionaries
- Database: Bulk creates PairEvent records
- Adapter: `PairCollectorAdapter`

#### 3. collector_holder
- Location: `modules/collector_holder/`
- Input: TokenInfo instance
- Output: List of holder dictionaries
- Database: Bulk creates HolderInfo records
- Adapter: `HolderCollectorAdapter`

#### 4. preprocessor
- Location: `modules/preprocessor/`
- Input: TokenInfo (with related pair_events and holders)
- Output:
  - Honeypot features dictionary
  - Exit features list (one dict per window)
- Database:
  - Creates HoneypotProcessedData record
  - Bulk creates ExitProcessedData records
- Adapter: `PreprocessorAdapter`

#### 5. honeypot_DA
- Location: `modules/honeypot_DA/`
- Input: TokenInfo, HoneypotProcessedData
- Output: Dictionary with detection results
- Database: None (results passed to aggregator)
- Adapter: `HoneypotDynamicAnalyzerAdapter`

#### 6. honeypot_ML
- Location: `modules/honeypot_ML/`
- Input: HoneypotProcessedData
- Output: Dictionary with ML predictions
- Database: None (results passed to aggregator)
- Adapter: `HoneypotMLAnalyzerAdapter`

#### 7. exit_ML
- Location: `modules/exit_ML/`
- Input: TokenInfo (accesses exit_processed relation)
- Output: Dictionary with ML predictions
- Database: None (results passed to aggregator)
- Adapter: `ExitMLAnalyzerAdapter`

#### 8. Aggregator
- Location: Built into pipeline
- Input: All analysis results
- Output: Dictionary with final risk score, scam types, insights
- Database: Creates Result record
- Adapter: `ResultAggregatorAdapter`

## API Layer

### Internal API (TBD)
- Path: `/api/`
- Purpose: Job submission and monitoring
- Endpoints:
  - POST `/api/jobs/` - Submit new analysis
  - GET `/api/jobs/{id}/` - Get job details
  - GET `/api/jobs/{id}/status/` - Get current status

### Frontend API (TBD)
Will be implemented to match existing frontend expectations:
- GET `/api/tokens/` - List analyzed tokens
- GET `/api/tokens/{address}/` - Get token details
- POST `/api/analyze/` - Start new analysis
- GET `/api/status/{job_id}/` - Poll analysis status

## File Structure

```
COINT/
├── api/
│   ├── models.py           # 7 database models
│   ├── serializers.py      # API serializers (minimal)
│   ├── views.py            # API views (minimal)
│   ├── urls.py             # URL routing
│   └── migrations/         # Database migrations
├── pipeline/
│   ├── adapters.py         # Module adapters with I/O specs
│   ├── orchestrator.py     # Pipeline execution coordinator
│   └── interfaces.py       # Base classes (legacy, may deprecate)
├── modules/
│   ├── collector_token/    # TODO: Integrate actual module
│   ├── collector_pair/     # TODO: Integrate actual module
│   ├── collector_holder/   # TODO: Integrate actual module
│   ├── preprocessor/       # TODO: Integrate actual module
│   ├── honeypot_DA/        # TODO: Integrate actual module
│   ├── honeypot_ML/        # TODO: Integrate actual module
│   └── exit_ML/            # TODO: Integrate actual module
├── config/
│   ├── settings.py         # Django configuration
│   └── urls.py             # Root URL configuration
└── frontend/               # React application (existing)
```

## Integration Checklist

To integrate actual analysis modules:

1. **For each collector module:**
   - Copy module code to `modules/{module_name}/`
   - Update adapter's `__init__()` to import actual module
   - Update adapter's `collect()` to call module functions
   - Ensure output format matches adapter specification
   - Convert any file I/O to in-memory data passing

2. **For preprocessor:**
   - Copy module code to `modules/preprocessor/`
   - Update adapter's `__init__()` to import actual module
   - Update `process_for_honeypot()` to call module
   - Update `process_for_exit()` to call module
   - Ensure output matches DB schema field names

3. **For ML/DA modules:**
   - Copy module code to `modules/{module_name}/`
   - Update adapter's `__init__()` to import and load models
   - Update prediction methods to call actual models
   - Ensure output format matches aggregator expectations

4. **For aggregator:**
   - Update `aggregate()` method with actual scoring logic
   - Define final risk score calculation formula
   - Define scam type identification rules
   - Define insight generation logic

## Next Steps

1. Test pipeline with sample data
2. Implement complete API endpoints for frontend
3. Create API serializers for Result model
4. Update frontend views to query result table first
5. Add async job execution (Celery integration)
6. Add authentication and rate limiting
7. Optimize database queries with proper indexing
8. Set up production deployment (PostgreSQL, Nginx, Gunicorn)

## Notes

- All database field names follow the provided specification exactly
- Timestamp fields accept ISO format strings (YYYY-MM-DDTHH:MM:SSZ)
- Decimal fields use max_digits=78, decimal_places=18 for Ethereum precision
- All adapters raise NotImplementedError until actual modules are integrated
- Pipeline automatically skips re-analysis if result already exists
- Job status is updated at each pipeline stage for progress tracking
