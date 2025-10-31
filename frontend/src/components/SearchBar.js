import { useState, useRef } from 'react';
import searchIcon from '../assets/icons/search.png';
import { useNavigate  } from 'react-router-dom';

function SearchBar() {  
    const [TokenAddr, setTokenAddr] = useState('');
    const inputRef = useRef(null);
    const navigate = useNavigate();

    const handleSearch = () => {
        const addr = (TokenAddr || '').trim();
        if (!addr) {
            navigate('/detail');
            return;
        } else {
            navigate(`/detail?address=${addr}`);
        }
        setTokenAddr('');
        inputRef.current?.blur();
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    return (
        <div style={{
            padding : '30px',
            color : '#171821',
            display: 'flex',
            gap: '10px'
        }}>
        <input
            ref={inputRef}
            value={TokenAddr}
            onChange={(e) => setTokenAddr(e.target.value)}
            onKeyDown={handleKeyPress}
            spellCheck={false}
            autoCapitalize="none"
            autoCorrect="off"
            placeholder="Search here.."
            style={{
                flex: '1',
                padding: '15px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: '#21222D',
                color: '#d2d2d2'
            }}
        />
        <button onClick={handleSearch}
            style={{
                padding: '10px 20px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: '#21222D',
                color: '#d2d2d2',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center'
            }}
        >
            <img
                src={searchIcon}
                alt="search"
                style={{
                    width: '15px',
                    height: '15px',
                    filter: 'brightness(0) invert(1)'
                }}
            />
        </button>

    </div>
    )

}

export default SearchBar;