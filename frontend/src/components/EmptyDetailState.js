import React from 'react';
import './EmptyDetailState.css';

function EmptyDetailState() {
    return (
        <div className="empty-detail-container">
            <div className="empty-detail-content">
                <div className="empty-detail-icon">
                    <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                        <circle cx="40" cy="40" r="38" stroke="#4880FF" strokeWidth="3" strokeDasharray="8 8"/>
                        <path d="M40 25V40L50 50" stroke="#4880FF" strokeWidth="3" strokeLinecap="round"/>
                    </svg>
                </div>
                <h2 className="empty-detail-title">검색할 토큰 주소를 입력해 주세요</h2>
                <p className="empty-detail-description">
                    상단 검색바에 토큰 주소를 입력하시면<br/>
                    해당 토큰의 상세 정보를 확인하실 수 있습니다.
                </p>
            </div>
        </div>
    );
}

export default EmptyDetailState;