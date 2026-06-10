import { useState, useRef, useEffect, useCallback } from 'react';
import { streamMessage, clearSession, fetchConfig, cancelSession } from './api';
import { exportChatToHtml } from './exportChat';
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';

export default function App() {
    // --- Persist session across page refreshes ---
    const [sessionId] = useState(() => {
        const saved = localStorage.getItem('cmip6-session-id');
        if (saved) return saved;
        const id = 'session_' + Math.random().toString(36).slice(2, 10);
        localStorage.setItem('cmip6-session-id', id);
        return id;
    });
    const [messages, setMessages] = useState(() => {
        try {
            const saved = localStorage.getItem('cmip6-messages');
            return saved ? JSON.parse(saved) : [];
        } catch { return []; }
    });
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [modelName, setModelName] = useState(() =>
        localStorage.getItem('cmip6-model') || 'gpt-5.2'
    );
    const [models, setModels] = useState([]);
    const [streamingText, setStreamingText] = useState('');
    const [streamingFigures, setStreamingFigures] = useState([]);
    const [streamingSources, setStreamingSources] = useState([]);
    const [agentStatus, setAgentStatus] = useState('');
    const [ragChunks, setRagChunks] = useState(() =>
        Number(localStorage.getItem('cmip6-rag-chunks')) || 25
    );
    const [ragSearches, setRagSearches] = useState(() =>
        Number(localStorage.getItem('cmip6-rag-searches')) || 12
    );
    const [reviewerModel1, setReviewerModel1] = useState(() => {
        const v = localStorage.getItem('cmip6-reviewer-model-1') || 'gemini-3.1-pro-preview';
        return (v === 'claude-opus-4-6' || v === 'claude-opus-4-7') ? 'claude-opus-4-8' : v;
    });
    const [reviewerModel2, setReviewerModel2] = useState(() => {
        const v = localStorage.getItem('cmip6-reviewer-model-2') || 'gemini-3.1-pro-preview';
        return (v === 'claude-opus-4-6' || v === 'claude-opus-4-7') ? 'claude-opus-4-8' : v;
    });
    const [reviewersEnabled, setReviewersEnabled] = useState(() =>
        localStorage.getItem('cmip6-reviewers-enabled') !== 'false'
    );
    const [reviewerModels, setReviewerModels] = useState([]);

    const chatEndRef = useRef(null);
    const inputRef = useRef(null);
    const abortControllerRef = useRef(null);

    // Persist messages to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem('cmip6-messages', JSON.stringify(messages));
    }, [messages]);

    // Persist settings
    useEffect(() => {
        localStorage.setItem('cmip6-model', modelName);
    }, [modelName]);
    useEffect(() => {
        localStorage.setItem('cmip6-rag-chunks', ragChunks);
    }, [ragChunks]);
    useEffect(() => {
        localStorage.setItem('cmip6-rag-searches', ragSearches);
    }, [ragSearches]);
    useEffect(() => {
        localStorage.setItem('cmip6-reviewer-model-1', reviewerModel1);
    }, [reviewerModel1]);
    useEffect(() => {
        localStorage.setItem('cmip6-reviewer-model-2', reviewerModel2);
    }, [reviewerModel2]);
    useEffect(() => {
        localStorage.setItem('cmip6-reviewers-enabled', reviewersEnabled);
    }, [reviewersEnabled]);


    // Load available models
    useEffect(() => {
        fetchConfig().then(cfg => {
            setModels(cfg.models || []);
            setModelName(cfg.current_model || 'gpt-5.2');
            if (cfg.reviewer_models) setReviewerModels(cfg.reviewer_models);

        }).catch(() => {
            setModels(['gpt-5.2', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-nano', 'gpt-4o-mini']);
            setReviewerModels(['gemini-3.1-pro-preview', 'claude-opus-4-8', 'gpt-5.5']);

        });
    }, []);

    // Restore theme from localStorage
    useEffect(() => {
        const saved = localStorage.getItem('cmip6-theme');
        if (saved === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    }, []);

    // Scroll to bottom on new messages
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, streamingText]);

    const handleSend = useCallback(async () => {
        const msg = input.trim();
        if (!msg || isLoading) return;

        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: msg }]);
        setIsLoading(true);
        setStreamingText('');
        setStreamingFigures([]);
        setStreamingSources([]);
        setAgentStatus('Thinking...');

        let fullText = '';
        let figures = [];
        let figCode = '';
        let figStdout = '';
        let allSources = [];

        // Set up an AbortController so the user can stop the stream client-side.
        abortControllerRef.current = new AbortController();

        await streamMessage(msg, sessionId, modelName, {
            signal: abortControllerRef.current.signal,
            ragChunks,
            ragSearches,
            reviewerModel1: reviewersEnabled ? reviewerModel1 : '',
            reviewerModel2: reviewersEnabled ? reviewerModel2 : '',
            reviewersEnabled,
            onText: (chunk) => {
                fullText += chunk;
                setStreamingText(fullText);
            },
            onStatus: (status) => {
                setAgentStatus(status);
            },
            onFigures: (paths, code, stdout) => {
                figures = [...paths];  // replace — show only latest version
                if (code) figCode = code;
                if (stdout) figStdout = stdout;
                setStreamingFigures([...figures]);
            },
            onSources: (query, results) => {
                allSources = [...allSources, { query, results }];
                setStreamingSources([...allSources]);
            },
            onDone: () => {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: fullText,
                    figure_paths: figures,
                    figure_code: figCode,
                    figure_stdout: figStdout,
                    sources: allSources,
                }]);
                setStreamingText('');
                setStreamingFigures([]);
                setStreamingSources([]);
                setAgentStatus('');
                setIsLoading(false);
            },
            onError: (err) => {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `⚠️ Error: ${err}`,
                }]);
                setStreamingText('');
                setStreamingFigures([]);
                setStreamingSources([]);
                setAgentStatus('');
                setIsLoading(false);
            },
        });
    }, [input, isLoading, sessionId, modelName, ragChunks, ragSearches, reviewerModel1, reviewerModel2, reviewersEnabled]);

    const handleStop = useCallback(async () => {
        // Tell the backend to break out at the next agent step boundary, then
        // abort the local fetch so the UI returns to idle immediately.
        if (sessionId) {
            cancelSession(sessionId);
        }
        try {
            abortControllerRef.current?.abort();
        } catch (_) { /* ignore */ }
        setAgentStatus('🛑 Stopping...');
    }, [sessionId]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleClearChat = async () => {
        await clearSession(sessionId);
        setMessages([]);
        setStreamingText('');
        setStreamingFigures([]);
        setStreamingSources([]);
        // Also clear persisted session and generate new one
        const newId = 'session_' + Math.random().toString(36).slice(2, 10);
        localStorage.setItem('cmip6-session-id', newId);
        localStorage.removeItem('cmip6-messages');
        // Reload to pick up the new session ID
        window.location.reload();
    };

    const handleExportChat = useCallback(() => {
        if (messages.length === 0) {
            alert('No messages to export');
            return;
        }
        exportChatToHtml(messages, modelName);
    }, [messages, modelName]);

    return (
        <div className="app">
            <Sidebar
                models={models}
                currentModel={modelName}
                onModelChange={setModelName}
                onClearChat={handleClearChat}
                onExportChat={handleExportChat}
                ragChunks={ragChunks}
                ragSearches={ragSearches}
                onRagChunksChange={setRagChunks}
                onRagSearchesChange={setRagSearches}
                reviewerModels={reviewerModels}
                reviewerModel1={reviewerModel1}
                reviewerModel2={reviewerModel2}
                onReviewerModel1Change={setReviewerModel1}
                onReviewerModel2Change={setReviewerModel2}
                reviewersEnabled={reviewersEnabled}
                onReviewersEnabledChange={setReviewersEnabled}

            />

            <main className="chat-main">
                <header className="chat-header">
                    <div className="chat-header-title">
                        <span className="logo-icon">🤖</span>
                        <h1>CMIP Forge</h1>
                    </div>
                    <span className="chat-header-subtitle">CMIP6 Climate Intelligence</span>
                </header>

                <div className="chat-messages">
                    {messages.length === 0 && !streamingText && (
                        <div className="empty-state">
                            <div className="empty-state-icon">🌐</div>
                            <h2>Welcome to CMIP Forge</h2>
                            <p>Ask me anything about CMIP6 climate data — search datasets, explore models, or analyze data.</p>
                            <div className="example-queries">
                                <button onClick={() => setInput('What is tos?')}>What is tos?</button>
                                <button onClick={() => setInput('Monthly sea surface temperature from MPI model, historical')}>
                                    Monthly SST from MPI, historical
                                </button>
                                <button onClick={() => setInput('Show me high resolution ocean models')}>
                                    High-res ocean models
                                </button>
                            </div>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <ChatMessage key={i} message={msg} />
                    ))}

                    {isLoading && !streamingText && agentStatus && (
                        <div className="message message-assistant">
                            <div className="message-header">
                                <div className="message-avatar">🤖</div>
                                <span className="message-role">CMIP Forge</span>
                            </div>
                            <div className="message-bubble">
                                <div className="agent-status">
                                    <span className="status-dot" />
                                    <span className="status-text">{agentStatus}</span>
                                </div>
                            </div>
                        </div>
                    )}

                    {streamingText && (
                        <ChatMessage
                            message={{
                                role: 'assistant',
                                content: streamingText,
                                figure_paths: streamingFigures,
                                sources: streamingSources,
                            }}
                            isStreaming={true}
                        />
                    )}

                    <div ref={chatEndRef} />
                </div>

                <div className="chat-input-area">
                    <div className="chat-input-wrapper">
                        <textarea
                            ref={inputRef}
                            className="chat-input"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask about CMIP6 climate data..."
                            rows={1}
                            disabled={isLoading}
                        />
                        {isLoading ? (
                            <button
                                className="send-button stop-button"
                                onClick={handleStop}
                                title="Stop the run"
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                                    <rect x="6" y="6" width="12" height="12" rx="2" />
                                </svg>
                            </button>
                        ) : (
                            <button
                                className="send-button"
                                onClick={handleSend}
                                disabled={!input.trim()}
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" />
                                </svg>
                            </button>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
