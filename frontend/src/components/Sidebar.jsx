export default function Sidebar({ models, currentModel, onModelChange, onClearChat }) {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <span className="logo-icon">🤖</span>
                <h2>CMIP Forge</h2>
            </div>

            <div className="sidebar-section">
                <label className="sidebar-label">Model</label>
                <select
                    className="sidebar-select"
                    value={currentModel}
                    onChange={(e) => onModelChange(e.target.value)}
                >
                    {models.map((m) => (
                        <option key={m} value={m}>{m}</option>
                    ))}
                </select>
            </div>

            <div className="sidebar-section">
                <label className="sidebar-label">Theme</label>
                <button
                    className="sidebar-button theme-toggle"
                    onClick={() => {
                        const html = document.documentElement;
                        const current = html.getAttribute('data-theme');
                        const next = current === 'light' ? 'dark' : 'light';
                        html.setAttribute('data-theme', next === 'dark' ? '' : 'light');
                        localStorage.setItem('cmip6-theme', next);
                    }}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="5" />
                        <line x1="12" y1="1" x2="12" y2="3" />
                        <line x1="12" y1="21" x2="12" y2="23" />
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                        <line x1="1" y1="12" x2="3" y2="12" />
                        <line x1="21" y1="12" x2="23" y2="12" />
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                    </svg>
                    Toggle Theme
                </button>
            </div>

            <div className="sidebar-spacer" />

            <button className="sidebar-button danger" onClick={onClearChat}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                Clear Chat
            </button>
        </aside>
    );
}
