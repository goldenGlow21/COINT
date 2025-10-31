import './TokenCard.css';

function TokenCard({token}) {

  // risk-score 색상 
  const getRiskColor = (score) => {
    if (score >= 80) return '#FF4444';
    if (score >= 60) return '#ff9500ff';
    if (score >= 30) return '#FFC107';
    return '#00C853';
  };

  return (
    <div className="token-card">
      <div className="token-info">
        <span className="token-name">{token.tokenName}</span>
        <span className="token-symbol">{token.Symbol}</span>
      </div>

      <div className="token-address">
        {token.address}
      </div>

      <div className="token-type">
        {token.Type}
      </div>

      <div className="risk-score"
      style={{color: getRiskColor(token.riskScore)}}
      >
        {token.riskScore}%
      </div>

    </div>
  );
}

export default TokenCard;