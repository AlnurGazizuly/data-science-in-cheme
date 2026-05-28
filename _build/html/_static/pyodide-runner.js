/**
 * Pyodide Inline Runner v5
 * - Pre-loads numpy, pandas, matplotlib, scipy via pyodide.loadPackage (bundled, reliable)
 * - Auto-injects np/pd/plt/sns aliases when used but not imported in the cell
 * - Handles both Jupyter notebook cells AND standalone Markdown code blocks
 * - Shared namespace across all cells on the page
 */

(function () {
  'use strict';

  const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js';

  // Packages to load via pyodide.loadPackage (bundled in Pyodide distribution)
  const LOAD_PACKAGES = ['numpy', 'pandas', 'matplotlib', 'scipy'];

  // Additional packages only available via micropip
  const MICROPIP_MAP = {
    'sklearn':   'scikit-learn',
    'cv2':       'opencv-python',
    'PIL':       'Pillow',
    'missingno': 'missingno',
    'shap':      'shap',
    'skopt':     'scikit-optimize',
    'imblearn':  'imbalanced-learn',
    'xgboost':   'xgboost',
    'plotly':    'plotly',
    'statsmodels': 'statsmodels',
    'nltk':      'nltk',
    'seaborn':   'seaborn',
    'joblib':    'joblib',
  };

  // Already covered by LOAD_PACKAGES — don't micropip these
  const PYODIDE_BUILTIN = new Set([
    'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'scikit-learn', 'PIL',
    'sqlite3', 'ssl', 'hashlib', 'hmac', 'lzma', 'bz2', 'zlib',
  ]);

  const BROWSER_INCOMPATIBLE = new Set([
    'streamlit','torch','gradio','langchain','tensorflow','keras',
    'flask','django','fastapi','celery','redis','psycopg2',
  ]);

  // Auto-inject these aliases if used but not imported in the cell
  const ALIAS_GUARDS = [
    { alias: 'np',  pattern: /\bnp\./,  inject: 'import numpy as np' },
    { alias: 'pd',  pattern: /\bpd\./,  inject: 'import pandas as pd' },
    { alias: 'plt', pattern: /\bplt\./, inject: 'import matplotlib.pyplot as plt' },
    { alias: 'sns', pattern: /\bsns\./, inject: 'import seaborn as sns' },
    { alias: 'sp',  pattern: /\bsp\./,  inject: 'import scipy as sp' },
  ];

  let pyodide = null;
  let pyodideLoading = false;
  let pyodideReady = false;
  const pendingRuns = [];
  let sharedNamespace = null;
  const installedPackages = new Set();

  // ── CSS ───────────────────────────────────────────────────────────────────
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
.pyodide-output.is-loading { color: rgba(255,255,255,0.45); font-style: italic; }
.pyodide-output img { max-width: 100%; margin-top: 6px; border-radius: 4px; display: block; }
.pyodide-banner {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 14px;
  background: rgba(243,139,168,0.08);
  border-top: 1px solid rgba(243,139,168,0.2);
  color: #f5c2e7;
  font-size: 0.82rem;
  line-height: 1.5;
}
.pyodide-banner .banner-icon { font-size: 1.1rem; flex-shrink: 0; }
.pyodide-banner code {
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
  padding: 1px 5px;
  font-family: monospace;
}
.pyodide-textarea:focus { background: rgba(255,255,255,0.03); }
`;

  function injectStyles() {
    if (document.getElementById('pyodide-runner-css')) return;
    const s = document.createElement('style');
    s.id = 'pyodide-runner-css';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  // ── Pyodide bootstrap — loads numpy/pandas/matplotlib/scipy up front ──────
  function loadPyodideIfNeeded() {
    if (pyodideReady || pyodideLoading) return;
    pyodideLoading = true;
    const script = document.createElement('script');
    script.src = PYODIDE_CDN;
    script.onload = async () => {
      try {
        pyodide = await loadPyodide();
        sharedNamespace = pyodide.globals;

        // Load bundled packages first — this is what fixes the numpy error
        await pyodide.loadPackage(LOAD_PACKAGES);

        // Setup capture + matplotlib backend
        await pyodide.runPythonAsync(`
import sys, io as _io
import matplotlib
matplotlib.use('Agg')
class _Capture(_io.StringIO): pass
_capture = _Capture()
sys.stdout = _capture
sys.stderr = _capture
_plot_images = []
`);
        // Load micropip for non-bundled packages
        await pyodide.loadPackage('micropip');

        pyodideReady = true;
        pyodideLoading = false;
        pendingRuns.forEach(fn => fn());
        pendingRuns.length = 0;
      } catch (e) {
        pyodideLoading = false;
        console.error('Pyodide failed to load:', e);
      }
    };
    script.onerror = () => { pyodideLoading = false; };
    document.head.appendChild(script);
  }

  function whenReady(fn) {
    if (pyodideReady) fn();
    else pendingRuns.push(fn);
  }

  // ── Parse explicit imports from code ─────────────────────────────────────
  function parseImports(code) {
    const imports = new Set();
    for (const line of code.split('\n')) {
      const m1 = line.match(/^\s*import\s+([\w,\s]+)/);
      const m2 = line.match(/^\s*from\s+(\w+)/);
      if (m1) m1[1].split(',').forEach(p => imports.add(p.trim().split(' ')[0]));
      if (m2) imports.add(m2[1]);
    }
    return imports;
  }

  function findIncompatible(imports) {
    return [...imports].filter(i => BROWSER_INCOMPATIBLE.has(i));
  }

  // Install packages only available via micropip (not bundled)
  async function ensureMicropipPackages(imports, outputEl) {
    const toInstall = [];
    for (const imp of imports) {
      if (PYODIDE_BUILTIN.has(imp) || installedPackages.has(imp)) continue;
      const pkg = MICROPIP_MAP[imp];
      if (pkg) toInstall.push({ imp, pkg });
    }
    if (!toInstall.length) return;
    for (const { imp, pkg } of toInstall) {
      outputEl.textContent = `⏳ Installing ${pkg}…`;
      try {
        await pyodide.runPythonAsync(`import micropip\nawait micropip.install('${pkg}')`);
        installedPackages.add(imp);
      } catch (e) {
        console.warn(`micropip install ${pkg} failed:`, e.message);
      }
    }
  }

  // Build silent preamble for aliases used but not imported
  function buildPreamble(code) {
    const lines = [];
    for (const { alias, pattern, inject } of ALIAS_GUARDS) {
      if (!pattern.test(code)) continue;
      const alreadyInCell = new RegExp(`(import\\s+\\S+\\s+as\\s+${alias}\\b|import\\s+${alias}\\b)`).test(code);
      if (alreadyInCell) continue;
      try {
        const inNS = pyodide.runPython(`'${alias}' in globals()`);
        if (inNS) continue;
      } catch(e) {}
      lines.push(inject);
    }
    return lines.join('\n');
  }

  // ── Run code ──────────────────────────────────────────────────────────────
  async function runCode(code, outputEl, bannerEl) {
    outputEl.className = 'pyodide-output has-content is-loading';
    outputEl.textContent = '⏳ Loading Python runtime…';
    bannerEl.style.display = 'none';

    const imports = parseImports(code);
    const incompatible = findIncompatible(imports);
    if (incompatible.length) {
      outputEl.className = 'pyodide-output';
      outputEl.textContent = '';
      bannerEl.style.display = 'flex';
      bannerEl.innerHTML = `
        <span class="banner-icon">⚠️</span>
        <div>
          <strong>Browser-incompatible library:</strong>
          ${incompatible.map(l => `<code>${l}</code>`).join(', ')} cannot run in the browser.<br>
          Run locally with <code>pip install ${incompatible.join(' ')}</code>.
        </div>`;
      return;
    }

    outputEl.textContent = '⏳ Starting Python…';

    // Only micropip for non-bundled packages
    await ensureMicropipPackages(imports, outputEl);

    // Auto-inject missing aliases
    const preamble = buildPreamble(code);
    const fullCode = preamble ? preamble + '\n' + code : code;

    // Reset capture
    await pyodide.runPythonAsync(`_capture.truncate(0); _capture.seek(0); _plot_images.clear()`);

    let textOut = '';
    let isError = false;
    let plotImages = [];

    try {
      const ret = await pyodide.runPythonAsync(fullCode, { globals: sharedNamespace });

      // Capture matplotlib figures
      if (/\bplt\./.test(code) || imports.has('matplotlib')) {
        try {
          await pyodide.runPythonAsync(`
import matplotlib.pyplot as plt, io as _mio, base64 as _mb64
for _fn in plt.get_fignums():
    _fig = plt.figure(_fn)
    _buf = _mio.BytesIO()
    _fig.savefig(_buf, format='png', bbox_inches='tight', dpi=120)
    _buf.seek(0)
    _plot_images.append(_mb64.b64encode(_buf.read()).decode())
    plt.close(_fig)
`);
          plotImages = pyodide.globals.get('_plot_images').toJs();
        } catch(e) {}
      }

      textOut = pyodide.runPython(`_capture.getvalue()`);
      if (ret !== undefined && ret !== null && String(ret) !== 'None') {
        if (textOut && !textOut.endsWith('\n')) textOut += '\n';
        textOut += String(ret);
      }
    } catch (err) {
      isError = true;
      const cap = pyodide.runPython(`_capture.getvalue()`);
      let msg = (cap || '') + (cap ? '\n' : '') + err.message;
      msg = msg.replace(/File "\/lib\/python[^"]*",\s*/g, '').replace(/\n\s*\^{5,}\n/g, '\n');
      if (preamble) {
        const offset = preamble.split('\n').length;
        msg = msg.replace(/line (\d+)/g, (m, n) => {
          const adj = parseInt(n) - offset;
          return adj > 0 ? `line ${adj}` : m;
        });
      }
      textOut = msg;
    }

    outputEl.className = 'pyodide-output has-content' + (isError ? ' is-error' : '');
    outputEl.textContent = '';
    if (textOut.trim()) {
      outputEl.textContent = textOut;
    } else if (!plotImages.length && !isError) {
      outputEl.textContent = '✓ Done (no output)';
    }
    for (const b64 of plotImages) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + b64;
      img.style.maxWidth = '100%';
      outputEl.appendChild(img);
    }
  }

  // ── Build editor widget ───────────────────────────────────────────────────
  function buildEditor(originalCode) {
    const wrap = document.createElement('div');
    wrap.className = 'pyodide-editor-wrap';

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

    const textarea = document.createElement('textarea');
    textarea.className = 'pyodide-textarea';
    textarea.value = originalCode;
    textarea.rows = Math.max(2, (originalCode.match(/\n/g) || []).length + 1);
    textarea.spellcheck = false;
    textarea.autocomplete = 'off';
    textarea.autocorrect = 'off';
    textarea.autocapitalize = 'off';
    textarea.addEventListener('input', () => {
      textarea.rows = Math.max(2, (textarea.value.match(/\n/g) || []).length + 1);
    });
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = textarea.selectionStart;
        textarea.value = textarea.value.slice(0, s) + '    ' + textarea.value.slice(textarea.selectionEnd);
        textarea.selectionStart = textarea.selectionEnd = s + 4;
      }
      if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); btn.click(); }
    });

    const output = document.createElement('div');
    output.className = 'pyodide-output';
    const banner = document.createElement('div');
    banner.className = 'pyodide-banner';
    banner.style.display = 'none';

    btn.addEventListener('click', () => {
      btn.disabled = true;
      const after = () => { btn.disabled = false; };
      if (!pyodideReady) {
        output.className = 'pyodide-output has-content is-loading';
        output.textContent = '⏳ Loading Python runtime (one-time, ~15s)…';
        loadPyodideIfNeeded();
        whenReady(() => runCode(textarea.value, output, banner).then(after).catch(after));
        return;
      }
      runCode(textarea.value, output, banner).then(after).catch(after);
    });

    wrap.appendChild(toolbar);
    wrap.appendChild(textarea);
    wrap.appendChild(banner);
    wrap.appendChild(output);
    return wrap;
  }

  // ── Extract plain text from a highlight div ───────────────────────────────
  function extractCode(el) {
    const pre = el.querySelector('pre');
    if (!pre) return '';
    return pre.textContent.replace(/^\n/, '').replace(/\n$/, '');
  }

  // ── Transform cells ───────────────────────────────────────────────────────
  function transformCells() {
    injectStyles();

    // 1. Jupyter notebook cells
    document.querySelectorAll('div.cell_input').forEach(cellInput => {
      const highlight = cellInput.querySelector('.highlight-python, .highlight-ipython3, .highlight-default');
      if (!highlight) return;
      const code = extractCode(highlight);
      if (!code.trim()) return;
      cellInput.querySelector('.cell_input div') && highlight.parentNode.replaceChild(buildEditor(code), highlight);
    });

    // 2. Standalone Markdown code blocks (not inside div.cell_input)
    const sel = [
      'div.highlight-python:not(.cell_input *)',
      'div.highlight-ipython3:not(.cell_input *)',
      'div.highlight-default:not(.cell_input *)',
      'div.highlight-shell:not(.cell_input *)',
    ].join(',');
    document.querySelectorAll(sel).forEach(el => {
      if (el.closest('.pyodide-editor-wrap')) return;
      const outer = el.closest('.highlight-python, .highlight-ipython3, .highlight-default, .highlight-shell') || el;
      const code = extractCode(outer);
      if (!code.trim()) return;
      outer.parentNode.replaceChild(buildEditor(code), outer);
    });

    // Preload Pyodide + packages in background immediately
    loadPyodideIfNeeded();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', transformCells);
  } else {
    transformCells();
  }
})();
