/**
 * API client for PangaeaGPT backend.
 */
const API_BASE = '/api';

export async function fetchConfig() {
    const res = await fetch(`${API_BASE}/config`);
    return res.json();
}

export async function createSession() {
    const res = await fetch(`${API_BASE}/sessions/create`, { method: 'POST' });
    return res.json();
}

export async function clearSession(sessionId) {
    const res = await fetch(`${API_BASE}/sessions/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    });
    return res.json();
}

export async function getMessages(sessionId) {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/messages`);
    return res.json();
}

/**
 * Send a chat message and get a full (non-streaming) response.
 */
export async function sendMessage(message, sessionId, modelName) {
    const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message,
            session_id: sessionId,
            model_name: modelName,
        }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || 'Request failed');
    }
    return res.json();
}

/**
 * Send a chat message and stream the response via SSE.
 * @param {string} message
 * @param {string} sessionId
 * @param {string} modelName
 * @param {function} onText - called with incremental text chunks
 * @param {function} onFigures - called with array of figure URLs
 * @param {function} onDone - called when stream completes
 * @param {function} onError - called on error
 */
export async function streamMessage(message, sessionId, modelName, { onText, onFigures, onDone, onError, onStatus, onSources }) {
    try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                session_id: sessionId,
                model_name: modelName,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            onError?.(err.detail || 'Request failed');
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        switch (data.type) {
                            case 'text':
                                onText?.(data.content);
                                break;
                            case 'status':
                                onStatus?.(data.content);
                                break;
                            case 'figures':
                                onFigures?.(data.paths, data.code, data.stdout);
                                break;
                            case 'sources':
                                onSources?.(data.query, data.results);
                                break;
                            case 'done':
                                onDone?.();
                                break;
                            case 'error':
                                onError?.(data.content);
                                break;
                        }
                    } catch (e) {
                        // skip unparseable lines
                    }
                }
            }
        }
    } catch (e) {
        onError?.(e.message);
    }
}
