import './ScamTypeCard.css';

function ScamTypeCard({ token }) {

  const getLevelColor = (level) => {
    switch(level) {
      case 'Critical': return '#FF4444';  // 빨강
      case 'Warning': return '#ff9500ff';   // 주황
      case 'Caution': return '#FFC107';   // 노랑
      case 'Normal': return '#00C853';    // 초록
      default: return '#FFFFFF';
    }
  }

  return (
    <div className="scam-type-card">
      <h3 className="card-title">Scam Type</h3>

      {token.scamTypes.map((item, index) => (
        <div className="scam-row" key={index}>
          <span className="scam-name">{item.type}</span>
          <span 
            className="scam-level"
            style={{ color: getLevelColor(item.level) }}
            >{item.level}</span>
            </div>
      ))}
    </div>
  )
}

export default ScamTypeCard;