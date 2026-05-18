/**
 * Export the current chat as a self-contained HTML file.
 * - Inlines all CSS (computed styles from the theme)
 * - Converts all images to base64 data URIs
 * - Preserves DOI links and external links
 * - Produces a single .html that can be emailed/shared
 */

const API_BASE = '/api';

/**
 * Convert an image URL to a base64 data URI.
 */
async function imageToBase64(url) {
    try {
        // For relative URLs (our figure API), make absolute
        const absoluteUrl = url.startsWith('/') ? window.location.origin + url : url;
        const res = await fetch(absoluteUrl);
        const blob = await res.blob();
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
            reader.onerror = () => resolve(url); // fallback to original URL
        });
    } catch {
        return url; // fallback
    }
}

/**
 * Build a self-contained HTML string from the messages array.
 */
export async function exportChatToHtml(messages, modelName = '') {
    // Dynamically import marked for markdown rendering
    // We'll do a simpler approach: clone the rendered DOM

    const chatContainer = document.querySelector('.chat-messages');
    if (!chatContainer) {
        alert('No chat to export');
        return;
    }

    // Clone the chat DOM
    const clone = chatContainer.cloneNode(true);

    // Remove streaming cursors and status indicators
    clone.querySelectorAll('.agent-status, .streaming-cursor').forEach(el => el.remove());

    // Convert all images to base64
    const images = clone.querySelectorAll('img');
    for (const img of images) {
        const src = img.getAttribute('src');
        if (src) {
            const dataUri = await imageToBase64(src);
            img.setAttribute('src', dataUri);
        }
    }

    // Grab computed CSS variables from the current theme
    const computedStyle = getComputedStyle(document.documentElement);
    const cssVars = [
        'bg-primary', 'bg-secondary', 'bg-tertiary', 'bg-hover',
        'bg-user-msg', 'bg-assistant-msg', 'bg-input',
        'border', 'border-focus',
        'text-primary', 'text-secondary', 'text-muted',
        'accent', 'accent-hover', 'accent-glow',
        'success', 'error',
        'code-bg', 'code-border',
        'inline-code-bg', 'inline-code-color',
        'link-color',
        'table-row-alt', 'table-row-hover',
        'radius', 'radius-sm', 'radius-xs',
        'font-sans', 'font-mono',
        'shadow-sm', 'shadow-md', 'shadow-lg',
    ].map(v => `--${v}: ${computedStyle.getPropertyValue(`--${v}`)};`).join('\n  ');

    // Collect all stylesheet rules (inline them)
    let allCSS = '';
    for (const sheet of document.styleSheets) {
        try {
            for (const rule of sheet.cssRules) {
                allCSS += rule.cssText + '\n';
            }
        } catch {
            // cross-origin sheets — skip
        }
    }

    const timestamp = new Date().toLocaleString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CMIP Forge — Chat Export (${timestamp})</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    ${cssVars}
  }

  ${allCSS}

  /* ── Export-specific overrides ── */
  body {
    overflow: auto !important;
    height: auto !important;
    padding: 0;
    margin: 0;
  }
  #root, .app, .chat-main {
    height: auto !important;
    overflow: visible !important;
  }
  .chat-messages {
    overflow: visible !important;
    height: auto !important;
    padding: 0 !important;
  }
  .sidebar, .chat-input-area, .chat-header,
  .copy-button, .figure-action-btn, .view-code-toggle,
  .sources-toggle svg.sources-chevron {
    /* hide interactive elements */
  }
  .export-header {
    text-align: center;
    padding: 32px 24px 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }
  .export-header h1 {
    font-family: var(--font-sans);
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 6px;
    letter-spacing: -0.02em;
  }
  .export-header .export-meta {
    font-size: 13px;
    color: var(--text-muted);
    font-family: var(--font-sans);
  }
  .export-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 24px 48px;
  }
  /* Ensure sources are visible in export */
  .sources-content {
    display: block !important;
  }
  .sources-toggle {
    pointer-events: none;
  }
  /* Make figure images responsive */
  .figure-item img, img {
    max-width: 100% !important;
    height: auto !important;
  }
  /* Print styles */
  @media print {
    body { background: white; color: #1a1a1a; }
    .message { break-inside: avoid; }
    .figure-item { break-inside: avoid; }
  }
</style>
</head>
<body>
  <div class="export-header">
    <h1>🤖 CMIP Forge — Chat Export</h1>
    <div class="export-meta">
      ${modelName ? `Model: ${modelName} · ` : ''}${timestamp} · ${messages.filter(m => m.role === 'user').length} questions
    </div>
  </div>
  <div class="export-wrapper">
    ${clone.innerHTML}
  </div>
</body>
</html>`;

    // Trigger download
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const dateStr = new Date().toISOString().slice(0, 10);
    a.download = `cmip_forge_${dateStr}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
