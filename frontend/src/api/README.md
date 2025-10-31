# API Client Usage Guide

## Import

```javascript
import api from './api/client';
```

## Usage Examples

### 1. Get Token List (Home Page)

```javascript
// In Home.js
import { useState, useEffect } from 'react';
import api from '../api/client';

function Home() {
    const [tokens, setTokens] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const fetchTokens = async () => {
            setLoading(true);
            try {
                const data = await api.getTokens();
                setTokens(data);
            } catch (error) {
                console.error('Failed to fetch tokens:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchTokens();
    }, []);

    // ... rest of component
}
```

### 2. Get Token Detail (Detail Page)

```javascript
// In Detail.js
import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import api from '../api/client';

function Detail() {
    const [searchParams] = useSearchParams();
    const address = searchParams.get('address');

    const [tokenData, setTokenData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!address) return;

        const fetchTokenData = async () => {
            setLoading(true);
            setError(null);

            try {
                const data = await api.getTokenDetail(address);
                setTokenData(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchTokenData();
    }, [address]);

    // ... rest of component
}
```

### 3. Analyze New Token

```javascript
// In SearchBar or Analysis component
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';

function AnalysisForm() {
    const [address, setAddress] = useState('');
    const [analyzing, setAnalyzing] = useState(false);
    const navigate = useNavigate();

    const handleAnalyze = async () => {
        if (!address) return;

        setAnalyzing(true);
        try {
            const result = await api.analyzeToken(address);
            console.log('Analysis started:', result);

            // Poll for status
            pollStatus(result.job_id);
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Analysis failed: ' + error.message);
        } finally {
            setAnalyzing(false);
        }
    };

    const pollStatus = async (jobId) => {
        const interval = setInterval(async () => {
            try {
                const status = await api.getStatus(jobId);
                console.log('Status:', status);

                if (status.status === 'completed') {
                    clearInterval(interval);
                    // Navigate to detail page
                    navigate(`/detail?address=${status.address}`);
                } else if (status.status === 'failed') {
                    clearInterval(interval);
                    alert('Analysis failed: ' + status.error);
                }
            } catch (error) {
                console.error('Status check failed:', error);
                clearInterval(interval);
            }
        }, 2000); // Poll every 2 seconds
    };

    // ... rest of component
}
```

## API Methods Reference

### `api.getTokens()`

Get list of analyzed tokens.

**Returns:** `Promise<Array>`

```javascript
[
  {
    id: 1,
    tokenName: "BitDao",
    Symbol: "BDO",
    address: "0x...",
    Type: "Honeypot, Exit",
    riskScore: 92.5
  }
]
```

### `api.getTokenDetail(address)`

Get detailed information for a specific token.

**Parameters:**
- `address` (string): Contract address (0x...)

**Returns:** `Promise<Object>`

```javascript
{
  id: 1,
  address: "0x...",
  tokenName: "BitDao",
  symbol: "BDO",
  tokenType: "ERC-20",
  contractOwner: "0x...",
  pair: "0x...",
  riskScore: 92.5,
  holders: [...],
  scamTypes: [...],
  victimInsights: [...]
}
```

### `api.analyzeToken(address, name?)`

Start analysis for a new token.

**Parameters:**
- `address` (string): Contract address (0x...)
- `name` (string, optional): Token name

**Returns:** `Promise<Object>`

```javascript
{
  job_id: 123,
  status: "pending",
  message: "Analysis started",
  address: "0x..."
}
```

### `api.getStatus(jobId)`

Check analysis job status.

**Parameters:**
- `jobId` (number): Job ID from analyzeToken response

**Returns:** `Promise<Object>`

```javascript
{
  job_id: 123,
  status: "analyzing",  // pending, collecting, preprocessing, analyzing, detecting, completed, failed
  progress: 60,         // 0-100
  error: null,
  address: "0x..."
}
```

## Error Handling

All API methods throw errors on failure. Use try-catch:

```javascript
try {
    const data = await api.getTokens();
} catch (error) {
    console.error('API Error:', error.message);
    // Handle error (show message to user, etc.)
}
```

## Migration from Mock Data

### Before (using mock data):
```javascript
import { mockHomeTokens } from '../mockData/homeData';

useEffect(() => {
    setTokens(mockHomeTokens);
}, []);
```

### After (using API):
```javascript
import api from '../api/client';

useEffect(() => {
    api.getTokens()
        .then(setTokens)
        .catch(console.error);
}, []);
```
