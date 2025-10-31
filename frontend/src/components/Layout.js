import SearchBar from './SearchBar';
import Sidebar from './Sidebar';


function Layout({ children }) {
    return (
        <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#171821' }}>
            <Sidebar />
            <div style={{ flex: 1, backgroundColor: '#171821' }}>
                <SearchBar />
                <main style={{
                    padding: '20px',
                    backgroundColor: '#171821'
                }}>
                    {children} 
                </main>
            </div>
        </div>
    )
}

export default Layout;