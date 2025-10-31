import { Link, useLocation } from 'react-router-dom';
import logo from '../logo.svg'
import homeIcon from '../assets/icons/home.png';
import detailIcon from '../assets/icons/detail.png'; 
import { motion } from 'framer-motion'

function Sidebar() {
    const location = useLocation(); 

    return (
        <div style={{
            width: '200px',
            backgroundColor : '#171821',
            padding: '30px',
            minHeight: '100vh'
        }}>
            {/* 로고 */}
            <Link to="/" style={{ testDecoration: 'none' }}>
                <img src={logo} alt="COINT Logo" style={{ width: '100px', marginLeft: '20px', marginBottom: '40px' }} />
            </Link>

            {/* Home 메뉴 */}
            <Link to="/" style={{ textDecoration: 'none' }}>
                <div style={{
                    padding: '10px',
                    color: location.pathname === '/' ? '#171821' : '#87888C',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center', 
                    marginBottom: location.pathname === '/' ? '8px' : '4px',
                    position: 'relative', 
                    zIndex: 1
                }}>
                    {location.pathname === '/' && (
                        <motion.div
                            layoutId="highlight"
                            transition={{ duration: 0.15 }}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                backgroundColor: '#4880FF',
                                borderRadius: '8px',
                                zIndex: -1
                            }}
                        />
                    )}
                    <img
                        src={homeIcon}
                        alt="Home"
                        style={{
                            width: '20px',
                            height: '20px',
                            marginRight: '15px',
                            marginLeft: '12px',
                            filter: location.pathname === '/' ? 'brightness(0)' : 'invert(1)',
                            position: 'relative',
                            zIndex: 1
                        }}
                    />
                    Home
                </div>
            </Link>

            {/* Detail 메뉴 */}
            <Link to="/detail" style={{ textDecoration: 'none' }}>
                <div style={{
                    padding: '10px',
                    color: location.pathname === '/detail' ? '#171821' : '#87888C',
                    backgroundColor : location.pathname === '/detail' ? '#4880FF' : 'transparent',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    marginTop: location.pathname === '/detail' ? '8px' : '8px',
                    marginBottom: location.pathname === '/detail' ? '8px' : '8px',
                    position: 'relative', 
                    zIndex: 1
                }}>
                    {location.pathname === '/detail' && (
                        <motion.div
                            layoutId="highlight"
                            transition={{ duration: 0.15 }}
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                backgroundColor: '#4880FF',
                                borderRadius: '8px',
                                zIndex: -1
                            }}
                        />
                    )}
                    <img 
                        src={detailIcon}
                        alt="Detail"
                        style={{
                            width: '20px',
                            height: '20px',
                            marginRight: '15px',
                            marginLeft: '12px',
                            filter: location.pathname === '/detail' ? 'brightness(0)' : 'invert(1)',
                            position: 'relative',
                            zIndex: 1
                        }}
                    />
                    Detail 
                </div>
            </Link>

        </div>
    );
}

export default Sidebar; 