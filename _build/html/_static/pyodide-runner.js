/**
 * Pyodide Inline Runner
 * Transforms all Python code cells into live editable + runnable editors.
 * Shared namespace across all cells on the page (like a real Jupyter session).
 */

(function () {
  'use strict';

  // ── Config ────────────────────────────────────────────────────────────────
  const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js';

  // ── State ─────────────────────────────────────────────────────────────────
  let pyodide = null;
  let pyodideLoading = false;
  let pyodideReady = false;
  const pendingRuns = [];        // callbacks waiting for pyodide to be ready

  // Shared Python namespace (globals dict, set once pyodide loads)
  let sharedNamespace = null;

  // ── Styles ────────────────────────────────────────────────────────────────
  const CSS = `
.pyodide-editor-wrap {
  margin: 0.5rem 0 0.25rem;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.12);
  background: #1e1e2e;
  font-family: 'Fira Mono', 'Consolas', monospace;
}
.pyodide-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 10px;
  background: rgba(0,0,0,0.25);
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.pyodide-toolbar .py-label {
  font-size: 0.7rem;
  color: rgba(255,255,255,0.35);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  user-select: none;
}
.pyodide-run-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 12px;
  border: none;
  border-radius: 4px;
  background: #4f9cf9;
  color: #fff;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
  user-select: none;
}
.pyodide-run-btn:hover:not(:disabled) { background: #3a88e8; }
.pyodide-run-btn:disabled { background: #555; cursor: wait; }
.pyodide-run-btn svg { width: 11px; height: 11px; fill: #fff; }
.pyodide-textarea {
  display: block;
  width: 100%;
  box-sizing: border-box;
  min-height: 52px;
  padding: 10px 14px;
  background: transparent;
  color: #cdd6f4;
  font-family: inherit;
  font-size: 0.88rem;
  line-height: 1.55;
  border: none;
  outline: none;
  resize: vertical;
  tab-size: 4;
  white-space: pre;
  overflow-x: auto;
}
.pyodide-output {
  display: none;
  border-top: 1px solid rgba(255,255,255,0.08);
  padding: 8px 14px;
  min-height: 1.5em;
  font-family: inherit;
  font-size: 0.85rem;
  white-space: pre-wrap;
  word-break: break-word;
  color: #a6e3a1;
  background: rgba(0,0,0,0.18);
  line-height: 1.5;
}
.pyodide-output.has-content { display: block; }
.pyodide-output.is-error { color: #f38ba8; }
.pyodide-output.is-loading { color: rgba(255,255,255,0.4); font-style: italic; }
/* make textarea tab-friendly */
.pyodide-textarea:focus { background: rgba(255,255,255,0.03); }
`;

  function injectStyles() {
    if (document.getElementById('pyodide-runner-css')) return;
    const style = document.createElement('style');
    style.id = 'pyodide-runner-css';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  // ── Pyodide bootstrap ─────────────────────────────────────────────────────
  function loadPyodideIfNeeded() {
    if (pyodideReady || pyodideLoading) return;
    pyodideLoading = true;

    const script = document.createElement('script');
    script.src = PYODIDE_CDN;
    script.onload = async () => {
      try {
        pyodide = await loadPyodide();
        // Create one shared globals namespace for the whole page
        sharedNamespace = pyodide.globals;
        // Redirect stdout/stderr into our capture buffer
        await pyodide.runPythonAsync(`
import sys, io as _io
class _CaptureIO(_io.StringIO):
    pass
_capture = _CaptureIO()
sys.stdout = _capture
sys.stderr = _capture
`);
        pyodideReady = true;
        pyodideLoading = false;
        pendingRuns.forEach(fn => fn());
        pendingRuns.length = 0;
      } catch (e) {
        pyodideLoading = false;
        console.error('Pyodide failed to load:', e);
      }
    };
    script.onerror = () => {
      pyodideLoading = false;
      console.error('Could not load Pyodide from CDN.');
    };
    document.head.appendChild(script);
  }

  function whenReady(fn) {
    if (pyodideReady) fn();
    else pendingRuns.push(fn);
  }

  // ── Run code ──────────────────────────────────────────────────────────────
  async function runCode(code, outputEl) {
    outputEl.className = 'pyodide-output has-content is-loading';
    outputEl.textContent = '⏳ Running…';

    // Reset capture buffer
    await pyodide.runPythonAsync(`_capture.truncate(0); _capture.seek(0)`);

    let result = '';
    let isError = false;

    try {
      const ret = await pyodide.runPythonAsync(code, { globals: sharedNamespace });
      result = pyodide.runPython(`_capture.getvalue()`);
      // If the last expression returned something meaningful, append it
      if (ret !== undefined && ret !== null) {
        const repr = String(ret);
        if (result && !result.endsWith('\n')) result += '\n';
        result += repr;
      }
    } catch (err) {
      isError = true;
      const captured = pyodide.runPython(`_capture.getvalue()`);
      result = captured ? captured + '\n' + err.message : err.message;
    }

    outputEl.className = 'pyodide-output has-content' + (isError ? ' is-error' : '');
    outputEl.textContent = result || (isError ? '(error with no message)' : '✓ Done (no output)');
  }

  // ── Build editor widget ───────────────────────────────────────────────────
  function buildEditor(originalCode, cellEl) {
    const wrap = document.createElement('div');
    wrap.className = 'pyodide-editor-wrap';

    // Toolbar
    const toolbar = document.createElement('div');
    toolbar.className = 'pyodide-toolbar';

    const label = document.createElement('span');
    label.className = 'py-label';
    label.textContent = 'Python';

    const btn = document.createElement('button');
    btn.className = 'pyodide-run-btn';
    btn.innerHTML = `<svg viewBox="0 0 384 512" xmlns="http://www.w3.org/2000/svg"><path d="M73 39c-14.8-9.1-33.4-9.4-48.5-.9S0 62.6 0 80L0 432c0 17.4 9.4 33.4 24.5 41.9s33.7 8.1 48.5-.9L361 271c14.3-8.7 23-24.2 23-41s-8.7-32.2-23-41L73 39z"/></svg> Run`;
    btn.title = 'Run cell (Shift+Enter)';

    toolbar.appendChild(label);
    toolbar.appendChild(btn);

    // Textarea
    const textarea = document.createElement('textarea');
    textarea.className = 'pyodide-textarea';
    textarea.value = originalCode;
    textarea.rows = Math.max(2, (originalCode.match(/\n/g) || []).length + 1);
    textarea.spellcheck = false;
    textarea.autocomplete = 'off';
    textarea.autocorrect = 'off';
    textarea.autocapitalize = 'off';

    // Auto-grow textarea
    textarea.addEventListener('input', () => {
      textarea.rows = Math.max(2, (textarea.value.match(/\n/g) || []).length + 1);
    });

    // Tab key support
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = textarea.selectionStart, end = textarea.selectionEnd;
        textarea.value = textarea.value.slice(0, s) + '    ' + textarea.value.slice(end);
        textarea.selectionStart = textarea.selectionEnd = s + 4;
      }
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        btn.click();
      }
    });

    // Output area
    const output = document.createElement('div');
    output.className = 'pyodide-output';

    // Run button click
    btn.addEventListener('click', () => {
      if (!pyodideReady) {
        output.className = 'pyodide-output has-content is-loading';
        output.textContent = '⏳ Loading Python runtime (first time only)…';
        loadPyodideIfNeeded();
        whenReady(() => runCode(textarea.value.trim(), output));
        return;
      }
      runCode(textarea.value.trim(), output);
    });

    wrap.appendChild(toolbar);
    wrap.appendChild(textarea);
    wrap.appendChild(output);
    return wrap;
  }

  // ── Extract raw code from a highlight div ─────────────────────────────────
  function extractCode(highlightDiv) {
    const pre = highlightDiv.querySelector('pre');
    if (!pre) return '';
    // textContent gives us the raw code without span tags
    return pre.textContent.replace(/^\n/, '').replace(/\n$/, '');
  }

  // ── Main transform ────────────────────────────────────────────────────────
  function transformCells() {
    injectStyles();

    // Target: div.cell_input containing a highlight-python block
    const cellInputs = document.querySelectorAll('div.cell_input');
    if (!cellInputs.length) return;

    cellInputs.forEach(cellInput => {
      const highlightDiv = cellInput.querySelector('.highlight-python');
      if (!highlightDiv) return;

      const code = extractCode(highlightDiv);
      if (!code) return;

      // Build the editor
      const editor = buildEditor(code, cellInput);

      // Replace the static code block with the editor
      highlightDiv.parentNode.replaceChild(editor, highlightDiv);
    });

    // Trigger Pyodide load in the background so it's ready when user first clicks
    loadPyodideIfNeeded();
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', transformCells);
  } else {
    transformCells();
  }
})();
