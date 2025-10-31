/**
 * API client configuration.
 * Centralized axios instance for backend communication.
 */

// Use native fetch for simplicity (no axios dependency needed)
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

/**
 * Fetch wrapper with error handling.
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;

    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    };

    try {
        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({
                error: `HTTP ${response.status}: ${response.statusText}`
            }));
            throw new Error(error.error || 'Request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

/**
 * API methods.
 */
export const api = {
    /**
     * Get list of analyzed tokens.
     */
    getTokens: () => apiRequest('/tokens/'),

    /**
     * Get token detail by address.
     */
    getTokenDetail: (address) => apiRequest(`/tokens/${address}/`),

    /**
     * Start new token analysis.
     */
    analyzeToken: (address, name = '') => apiRequest('/analyze/', {
        method: 'POST',
        body: JSON.stringify({ address, name }),
    }),

    /**
     * Check analysis status.
     */
    getStatus: (jobId) => apiRequest(`/status/${jobId}/`),
};

export default api;
