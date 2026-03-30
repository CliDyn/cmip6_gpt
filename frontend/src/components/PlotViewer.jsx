import { useState, useCallback } from 'react';

export default function PlotViewer({ paths, code, stdout }) {
    const [expandedIdx, setExpandedIdx] = useState(null);
    const [showCode, setShowCode] = useState(false);
    const [copied, setCopied] = useState(false);

    if (!paths || paths.length === 0) return null;

    const handleDownload = (path, idx) => {
        const link = document.createElement('a');
        link.href = path;
        link.download = `plot_${idx + 1}.png`;
        link.click();
    };

    const handleCopyCode = useCallback(() => {
        if (code) {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    }, [code]);

    return (
        <>
            <div className="figure-gallery">
                {paths.map((path, i) => (
                    <div key={i} className="figure-item">
                        <img
                            src={path}
                            alt={`Generated plot ${i + 1}`}
                            loading="lazy"
                            onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                            style={{ cursor: 'pointer' }}
                        />
                        <div className="figure-actions">
                            <button
                                className="figure-action-btn"
                                onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                                title="Expand"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <polyline points="15 3 21 3 21 9" />
                                    <polyline points="9 21 3 21 3 15" />
                                    <line x1="21" y1="3" x2="14" y2="10" />
                                    <line x1="3" y1="21" x2="10" y2="14" />
                                </svg>
                                Expand
                            </button>
                            <button
                                className="figure-action-btn"
                                onClick={() => handleDownload(path, i)}
                                title="Download"
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                    <polyline points="7 10 12 15 17 10" />
                                    <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                                Download
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            {/* View Code toggle */}
            {code && (
                <div className="view-code-section">
                    <button
                        className="view-code-toggle"
                        onClick={() => setShowCode(!showCode)}
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="16 18 22 12 16 6" />
                            <polyline points="8 6 2 12 8 18" />
                        </svg>
                        {showCode ? 'Hide code' : 'View code'}
                        <svg
                            width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                            style={{ transform: showCode ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }}
                        >
                            <polyline points="6 9 12 15 18 9" />
                        </svg>
                    </button>
                    {showCode && (
                        <div className="code-block-wrapper">
                            <div className="code-block-header">
                                <span className="code-block-lang">python</span>
                                <button className={`copy-button ${copied ? 'copied' : ''}`} onClick={handleCopyCode}>
                                    {copied ? '✓ Copied' : 'Copy'}
                                </button>
                            </div>
                            <pre className="language-python"><code>{code}</code></pre>
                        </div>
                    )}
                    {showCode && stdout && (
                        <div className="code-stdout">
                            <span className="code-stdout-label">Output</span>
                            <pre>{stdout}</pre>
                        </div>
                    )}
                </div>
            )}

            {/* Lightbox */}
            {expandedIdx !== null && (
                <div
                    className="figure-lightbox"
                    onClick={() => setExpandedIdx(null)}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0, 0, 0, 0.85)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 1000,
                        cursor: 'zoom-out',
                        animation: 'fadeIn 0.2s ease',
                    }}
                >
                    <img
                        src={paths[expandedIdx]}
                        alt={`Plot ${expandedIdx + 1} (expanded)`}
                        style={{
                            maxWidth: '90vw',
                            maxHeight: '90vh',
                            borderRadius: '8px',
                            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                        }}
                    />
                </div>
            )}
        </>
    );
}
