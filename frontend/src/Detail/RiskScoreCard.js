import './RiskScoreCard.css';

function RiskScoreCard({ token }) {
    const radius = 70;
    const circumference = Math.PI * radius;
    const offset = circumference - (token.riskScore / 100 ) * circumference;

    const getRiskColor = (score) => {
        if (score >= 80) return '#FF4444';
        if (score >= 60) return '#ff9500ff';
        if (score >= 30) return '#FFC107';
        return '#00c853';
    };

    return (
        <div className="risk-score-card">
            <h3 className="card-title">Risk Score</h3>

            <div className="circle-container">
                <svg width="180" height="100" viewBox="0 0 180 100">

                    <path
                        d={`M 20,90 A ${radius},${radius} 0 0,1 160,90`}
                        fill="none"
                        stroke="#2a2d33"
                        strokeWidth="20"
                    />

                    <path
                        d={`M 20,90 A ${radius},${radius} 0 0,1 160,90`}
                        fill="none"
                        stroke={getRiskColor(token.riskScore)}
                        strokeWidth="20"
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                    />
                </svg>
                <div className="score-text">{token.riskScore}%</div>
            </div>
        </div>
    )
}

export default RiskScoreCard;