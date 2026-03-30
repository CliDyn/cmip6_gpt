import { useState, useRef, useEffect, useCallback } from 'react';
import { streamMessage, clearSession, fetchConfig } from './api';
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';

export default function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId] = useState(() => 'session_' + Math.random().toString(36).slice(2, 10));
    const [modelName, setModelName] = useState('gpt-5.2');
    const [models, setModels] = useState([]);
    const [streamingText, setStreamingText] = useState('');
    const [streamingFigures, setStreamingFigures] = useState([]);
    const [streamingSources, setStreamingSources] = useState([]);
    const [agentStatus, setAgentStatus] = useState('');
    const chatEndRef = useRef(null);
    const inputRef = useRef(null);

    // Load available models
    useEffect(() => {
        fetchConfig().then(cfg => {
            setModels(cfg.models || []);
            setModelName(cfg.current_model || 'gpt-5.2');
        }).catch(() => {
            setModels(['gpt-5.2', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-nano', 'gpt-4o-mini']);
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

        await streamMessage(msg, sessionId, modelName, {
            onText: (chunk) => {
                fullText += chunk;
                setStreamingText(fullText);
            },
            onStatus: (status) => {
                setAgentStatus(status);
            },
            onFigures: (paths, code, stdout) => {
                figures = [...figures, ...paths];
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
    }, [input, isLoading, sessionId, modelName]);

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
    };

    return (
        <div className="app">
            <Sidebar
                models={models}
                currentModel={modelName}
                onModelChange={setModelName}
                onClearChat={handleClearChat}
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
                        <button
                            className="send-button"
                            onClick={handleSend}
                            disabled={isLoading || !input.trim()}
                        >
                            {isLoading ? (
                                <span className="spinner" />
                            ) : (
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" />
                                </svg>
                            )}
                        </button>
                    </div>
                </div>
            </main>
        </div>
    );
}
