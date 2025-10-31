export default function VictimInsightsCard({ items = [] }) {
    if (!items.length) {
        return <div style={{color: '#87888c'}}>탐지 지표가 없습니다.</div>
    }
    return (
        <div>
            <div className="victim-header">
                <h3>Victim Insights</h3>
                <span className="victim-count">탐지 지표: {items.length}개</span>
            </div>
            <ul className="insight-list">
                {items.map((v, i) => (
                    <li key={i} className="insight-item">
                        <strong className="insight-title">{v.title}</strong>
                        <p className="insight-desc">{v.description}</p>
                    </li>
                ))}
            </ul>
        </div>
    );
}