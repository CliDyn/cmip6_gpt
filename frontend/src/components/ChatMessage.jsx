import { useState, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import PlotViewer from './PlotViewer';

function CodeBlock({ children, className }) {
    const [copied, setCopied] = useState(false);
    const code = String(children).replace(/\n$/, '');

    // Extract language from className (e.g., "language-python" → "python")
    const lang = className?.replace('language-', '') || '';

    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    }, [code]);

    return (
        <div className="code-block-wrapper">
            <div className="code-block-header">
                <span className="code-block-lang">{lang || 'code'}</span>
                <button className={`copy-button ${copied ? 'copied' : ''}`} onClick={handleCopy}>
                    {copied ? (
                        <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12" /></svg> Copied</>
                    ) : (
                        <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></svg> Copy</>
                    )}
                </button>
            </div>
            <pre className={className}>
                <code>{code}</code>
            </pre>
        </div>
    );
}

/**
 * Convert DOI patterns in text to clickable markdown links.
 * Two-phase approach to prevent double-linking:
 *   Phase 1: Unwrap any markdown DOI links the agent already wrote → bare DOIs
 *   Phase 2: Uniformly wrap ALL bare DOIs as [DOI](https://doi.org/DOI)
 */
function linkifyDois(text) {
    if (!text) return '';

    // Phase 1: Unwrap agent-written DOI markdown links to bare DOIs.
    // Matches: [10.xxx/yyy](https://doi.org/10.xxx/yyy) → 10.xxx/yyy
    // Also: [DOI: 10.xxx](url) variants
    let result = text.replace(
        /\[(?:DOI:\s*)?(10\.\d{4,9}\/[^\]]+?)\]\(https?:\/\/doi\.org\/[^)]+\)/gi,
        (_, doi) => doi.replace(/[.),;]+$/, '')
    );

    // Phase 2: Wrap ALL bare DOIs that aren't already inside a markdown link.
    // Skip DOIs inside URLs (preceded by / from https://doi.org/...)
    result = result.replace(
        /(?<![([\/])(10\.\d{4,9}\/[^\s)\],;]+)/g,
        (match, doi) => {
            const cleanDoi = doi.replace(/[.),;]+$/, '');
            const trailing = doi.slice(cleanDoi.length);
            return `[${cleanDoi}](https://doi.org/${cleanDoi})${trailing}`;
        }
    );

    return result;
}

export default function ChatMessage({ message, isStreaming = false }) {
    const { role, content, figure_paths, figure_code, figure_stdout, sources } = message;
    const isUser = role === 'user';
    const [sourcesOpen, setSourcesOpen] = useState(false);

    // Count total unique papers across all search queries
    const totalSources = useMemo(() => {
        if (!sources || sources.length === 0) return 0;
        const seen = new Set();
        sources.forEach(s => s.results?.forEach(r => seen.add(r.doi || r.title)));
        return seen.size;
    }, [sources]);

    // Memoize DOI linkification so it doesn't re-run on every render
    const processedContent = useMemo(() => {
        if (isUser || !content) return content || '';
        return linkifyDois(content);
    }, [content, isUser]);

    return (
        <div className={`message message-${role}`}>
            <div className="message-header">
                <div className="message-avatar">
                    {isUser ? '👤' : '🤖'}
                </div>
                <span className="message-role">{isUser ? 'You' : 'CMIP Forge'}</span>
            </div>
            <div className="message-bubble">
                <div className={`message-content ${isStreaming ? 'streaming-cursor' : ''}`}>
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                        components={{
                            code({ node, inline, className, children, ...props }) {
                                const code = String(children).replace(/\n$/, '');
                                const hasLang = className && className.startsWith('language-');
                                const isShortSingleLine = !code.includes('\n') && code.length <= 80;

                                if (!inline && isShortSingleLine && !hasLang) {
                                    return <code className={className} {...props}>{children}</code>;
                                }
                                if (!inline) {
                                    return <CodeBlock className={className}>{children}</CodeBlock>;
                                }
                                return <code className={className} {...props}>{children}</code>;
                            },
                            a({ href, children }) {
                                const isExternal = href?.startsWith('http');
                                const isDoi = href?.startsWith('https://doi.org/');
                                return (
                                    <a
                                        href={href}
                                        target={isExternal ? "_blank" : undefined}
                                        rel={isExternal ? "noopener noreferrer" : undefined}
                                        className={`${isExternal ? "external-link" : ""} ${isDoi ? "doi-link" : ""}`}
                                    >
                                        {isDoi && (
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ marginRight: 3, verticalAlign: 'middle', opacity: 0.7 }}>
                                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                                            </svg>
                                        )}
                                        {children}
                                        {isExternal && !isDoi && (
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginLeft: 4, verticalAlign: 'middle' }}>
                                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                                                <polyline points="15 3 21 3 21 9" />
                                                <line x1="10" y1="14" x2="21" y2="3" />
                                            </svg>
                                        )}
                                    </a>
                                );
                            },
                            table({ children }) {
                                return (
                                    <div className="table-wrapper">
                                        <table>{children}</table>
                                    </div>
                                );
                            },
                            blockquote({ children }) {
                                return <blockquote className="callout">{children}</blockquote>;
                            },
                            h2({ children }) {
                                return <h2 className="section-heading">{children}</h2>;
                            },
                            h3({ children }) {
                                return <h3 className="section-heading">{children}</h3>;
                            },
                        }}
                    >
                        {processedContent}
                    </ReactMarkdown>
                </div>

                {figure_paths && figure_paths.length > 0 && (
                    <PlotViewer paths={figure_paths} code={figure_code} stdout={figure_stdout} />
                )}

                {/* ── RAG Sources Panel ── */}
                {totalSources > 0 && (
                    <div className="sources-panel">
                        <button
                            className={`sources-toggle ${sourcesOpen ? 'open' : ''}`}
                            onClick={() => setSourcesOpen(!sourcesOpen)}
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                            </svg>
                            <span>{totalSources} sources retrieved</span>
                            <span className="sources-queries">{sources.length} {sources.length === 1 ? 'search' : 'searches'}</span>
                            <svg className="sources-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="6 9 12 15 18 9" />
                            </svg>
                        </button>
                        {sourcesOpen && (
                            <div className="sources-content">
                                {sources.map((searchGroup, gi) => (
                                    <div key={gi} className="sources-group">
                                        <div className="sources-group-query">
                                            <span className="sources-query-icon">🔍</span>
                                            <span className="sources-query-text">{searchGroup.query}</span>
                                        </div>
                                        {searchGroup.results?.map((src, si) => (
                                            <div key={si} className="source-card">
                                                <div className="source-header">
                                                    <span className="source-rank">#{si + 1}</span>
                                                    <span className="source-title">{src.title}</span>
                                                    <span className="source-score">{(src.score * 100).toFixed(0)}%</span>
                                                </div>
                                                <div className="source-meta">
                                                    {src.year && <span className="source-year">{src.year}</span>}
                                                    {src.journal && <span className="source-journal">{src.journal}</span>}
                                                    {src.doi && (
                                                        <a
                                                            href={`https://doi.org/${src.doi}`}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            className="doi-link source-doi"
                                                        >
                                                            {src.doi}
                                                        </a>
                                                    )}
                                                </div>
                                                {src.text && (
                                                    <div className="source-text">{src.text}</div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
