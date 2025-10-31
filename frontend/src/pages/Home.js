import { useState, useEffect } from 'react';
import { mockHomeTokens } from '../mockData/homeData';
import TokenCard from '../Home/TokenCard';
import './Home.css';

function Home() {
    // State 정의 
    const [tokens, setTokens] = useState([]);
    const [loading, setLoading] = useState(false);
    const [currentPage, setCurrentPage] = useState(1);
    const tokensPerPage = 10;

    // token(Home) API 호출 
    useEffect(() => {
        setLoading(true);
        setTimeout(() => {
            setTokens(mockHomeTokens);
            setLoading(false);
        }, 100);
    }, []);

    // 페이지네이션 
    const indexOfLastToken = currentPage * tokensPerPage;
    const indexOfFirstToken = indexOfLastToken - tokensPerPage;
    const currentTokens = tokens.slice(indexOfFirstToken, indexOfLastToken);
    const totalPages = Math.ceil(tokens.length / tokensPerPage);

    // 페이지 변경 
    const handlePageChange = (pageNumber) => {
        setCurrentPage(pageNumber);
    };

    return (
        <div style={{ padding: '10px' }}>
            {/*컬럼명 헤더*/}
            <div className="token-table-grid" style={{ color:'#D2D2D2', fontSize:'16px', fontWeight:'bold', marginBottom:'10px' }}>
                <span>Token Name(Symbol)</span>
                <span className="header-address">Token Address</span>
                <span className="header-type">Type</span>
                <span style={{justifySelf:'end'}} >Risk Score</span>
            </div>

            {/*토큰 카드*/}
            <div>
                {loading ? (
                    <div>Loading...</div>
                ) : currentTokens.length > 0 ? (
                    currentTokens.map((token) => (
                        <TokenCard
                        key={token.id}
                        token={token}
                        />
                    ))
                ) : (
                    <div style={{
                        textAlign: 'center',
                        color: '#87888C',
                        padding: '40px'
                    }}>
                        데이터 없음 
                    </div>
                )}
            </div>

            {/*페이지네이션*/}
            {!loading && tokens.length > 0 && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '8px',
                    marginTop: '30px'
                }}>
                    {Array.from({ length: totalPages }, (_, index) => (
                        <button 
                        key={index + 1}
                        onClick={() => handlePageChange(index + 1)}
                        style={{
                            padding: '10px 16px',
                            borderRadius: '8px',
                            border: 'none',
                            backgroundColor: currentPage === index + 1 ? '#4880FF' : '#21222d',
                            color: currentPage === index + 1 ? '#FFFFFF' : '#D2D2D2',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            t: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                            if (currentPage !== index +1 ) {
                                e.target.style.backgroundColor = '#2a2b38';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (currentPage !== index + 1 ) {
                                e.target.style.backgroundColor = '#21222d';
                            }
                        }}
                        >{index + 1} </button>
                    ))}
                    </div>
            )}
        </div>
    )

}

export default Home;