import './TokenInfoCard.css';
import { useState } from 'react';

function TokenInfoCard({ token }) {
  const [copiedKey, setCopiedKey] = useState(null);  // ← 이렇게 수정!

  const copyAddr = (key, value) => {  // ← 함수명 오타 수정 (conpyAddr → copyAddr)
    navigator.clipboard.writeText(value);
    setCopiedKey(key);  // ← 어떤 버튼인지 저장
    setTimeout(() => setCopiedKey(null), 900);  // ← 900ms 후 초기화
  }

  return (
    <div className="token-info-card">

      {/* 헤더 영역 */}
      <div className="card-header">
        <span className="card-title">Token Info</span>
      </div>
      
      {/* Token Name */}
      <div className="token-name-section">
        <span className="token-name">{token.tokenName}</span>
        <span className="token-symbol">{token.symbol}</span>
        <span className="addr-pill">{token.address}</span>
        <button className="icon-btn" onClick={() => copyAddr('token', token.address)}>
          {copiedKey === 'token' ? '✓' : '⧉'}
        </button>
      </div>

      <div className="info-item">
        <span className="info-label">Token Type</span>
        <span className="info-value">{token.tokenType}</span>
      </div>

      <div className="info-item">
        <span className="info-label">Contract Owner</span>
        <span className="info-value">{token.contractOwner}</span>
        <button className="icon-btn" onClick={() => copyAddr('owner', token.contractOwner)}>
          {copiedKey === 'owner' ? '✓' : '⧉'}
        </button>   
      </div>  

      <div className="info-item">
        <span className="info-label">Pair</span>
        <span className="info-value">{token.pair}</span>
        <button className="icon-btn" onClick={() => copyAddr('pair', token.pair)}>
          {copiedKey === 'pair' ? '✓' : '⧉'}
        </button>
      </div>        

    </div>
  );
}

export default TokenInfoCard;