/**
 * Pyodide Inline Runner v7
 * Fixes applied before execution:
 *  1. Pre-loads numpy/pandas/matplotlib/scipy via loadPackage
 *  2. Auto-injects np/pd/plt/sns aliases if used but not imported
 *  3. Rewrites pd.read_excel/read_csv to use a synthetic demo DataFrame
 *  4. Auto-defines `data` if used but not in scope
 *  5. Dedents code with excess leading whitespace
 *  6. Removes bare `return` statements outside any function
 *  7. Fixes unmatched open parentheses at end of print/function-call lines
 *  8. Handles both Jupyter cells and standalone Markdown code blocks
 *  9. Shared namespace across all cells on the page
 */

(function () {
  'use strict';

  const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js';
  const LOAD_PACKAGES = ['numpy', 'pandas', 'matplotlib', 'scipy'];

  const MICROPIP_MAP = {
    'sklearn':     'scikit-learn',
    'cv2':         'opencv-python',
    'PIL':         'Pillow',
    'missingno':   'missingno',
    'shap':        'shap',
    'skopt':       'scikit-optimize',
    'imblearn':    'imbalanced-learn',
    'xgboost':     'xgboost',
    'plotly':      'plotly',
    'statsmodels': 'statsmodels',
    'nltk':        'nltk',
    'seaborn':     'seaborn',
    'joblib':      'joblib',
  };

  const PYODIDE_BUILTIN = new Set([
    'numpy','pandas','matplotlib','scipy','sqlite3','ssl',
    'hashlib','hmac','lzma','bz2','zlib',
  ]);

  const BROWSER_INCOMPATIBLE = new Set([
    'streamlit','torch','gradio','langchain','tensorflow','keras',
    'flask','django','fastapi','celery','redis','psycopg2',
  ]);

  const ALIAS_GUARDS = [
    { alias: 'np',  pattern: /\bnp\./,  inject: 'import numpy as np' },
    { alias: 'pd',  pattern: /\bpd\./,  inject: 'import pandas as pd' },
    { alias: 'plt', pattern: /\bplt\./, inject: 'import matplotlib.pyplot as plt' },
    { alias: 'sns', pattern: /\bsns\./, inject: 'import seaborn as sns' },
    { alias: 'sp',  pattern: /\bsp\./,  inject: 'import scipy as sp' },
  ];

  // Demo DataFrame covering all column names used across sections 2 & 3
  const DEMO_DATA_PY = `
import pandas as pd
import numpy as np
_demo_data = pd.DataFrame({
    'Temperature':   [295, 305, 315, 298, 310, 320, 290, 302, 318, 308],
    'Pressure':      [1.1, 1.8, 2.5, 1.3, 2.1, 2.9, 0.9, 1.6, 2.7, 2.2],
    'FlowRate':      [10.2, 15.4, 20.1, 12.3, 17.8, 22.5, 9.8, 14.6, 21.3, 18.9],
    'Concentration': [0.5,  1.0,  1.5,  0.7,  1.2,  1.8,  0.4, 0.9,  1.6,  1.3],
    'sepal_length':  [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
    'sepal_width':   [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
    'petal_length':  [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
    'petal_width':   [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
    'species':       ['setosa']*10,
})
data = _demo_data.copy()
`;

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
.pyodide-note {
  padding: 4px 14px;
  font-size: 0.75rem;
  color: rgba(255,255,255,0.35);
  font-style: italic;
  background: rgba(0,0,0,0.12);
  border-top: 1px solid rgba(255,255,255,0.05);
}
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
  background: rgba(255,255,255,0.1); border-radius: 3px;
  padding: 1px 5px; font-family: monospace;
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

  // ── Pyodide bootstrap ─────────────────────────────────────────────────────
  function loadPyodideIfNeeded() {
    if (pyodideReady || pyodideLoading) return;
    pyodideLoading = true;
    const script = document.createElement('script');
    script.src = PYODIDE_CDN;
    script.onload = async () => {
      try {
        pyodide = await loadPyodide();
        sharedNamespace = pyodide.globals;
        await pyodide.loadPackage(LOAD_PACKAGES);
        await pyodide.runPythonAsync(`
import sys, io as _io
import matplotlib; matplotlib.use('Agg')
class _Capture(_io.StringIO): pass
_capture = _Capture()
sys.stdout = _capture
sys.stderr = _capture
_plot_images = []
`);
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

  // ── Code sanitizers ───────────────────────────────────────────────────────

  // 1. Dedent
  function dedent(code) {
    const lines = code.split('\n');
    const nonEmpty = lines.filter(l => l.trim().length > 0);
    if (!nonEmpty.length) return code;
    const minIndent = Math.min(...nonEmpty.map(l => l.match(/^(\s*)/)[1].length));
    return minIndent === 0 ? code : lines.map(l => l.slice(minIndent)).join('\n');
  }

  // 2. Remove bare `return expr` lines that are outside any function/class.
  //    These appear in cells that were copy-pasted from inside a function body.
  function removeOuterReturns(code) {
    const lines = code.split('\n');
    let indentDepth = 0;   // track def/class nesting
    const result = [];
    for (const line of lines) {
      const stripped = line.trim();
      // detect entering a function/class block
      if (/^(def |class |async def )/.test(stripped)) {
        indentDepth++;
      }
      // detect leaving (a non-empty, non-comment line at indent 0 after we were inside)
      if (indentDepth > 0 && line.length > 0 && !stripped.startsWith('#')) {
        const leadingSpaces = line.match(/^(\s*)/)[1].length;
        if (leadingSpaces === 0 && !(/^(def |class |async def )/.test(stripped))) {
          indentDepth = 0;
        }
      }
      // Drop bare `return` at top level (indentDepth === 0)
      if (indentDepth === 0 && /^\s*return\b/.test(line)) {
        // Convert to a comment so the intent is preserved but syntax is valid
        result.push(line.replace(/^\s*return\b/, '# return'));
      } else {
        result.push(line);
      }
    }
    return result.join('\n');
  }

  // 3. Fix unmatched open parentheses: a line where open parens > close parens
  //    at the END of the line (i.e., the closing paren is missing entirely).
  //    Only fix lines that look like function calls (print, etc.) at top level.
  function fixUnmatchedParens(code) {
    const lines = code.split('\n');
    const result = [];
    for (const line of lines) {
      const open = (line.match(/\(/g) || []).length;
      const close = (line.match(/\)/g) || []).length;
      const diff = open - close;
      if (diff > 0 && !line.trim().startsWith('#')) {
        // Only fix if the line ends with a quote or paren-like char
        // (i.e. it looks truncated, not a multi-line expression)
        const trimmed = line.trimEnd();
        if (/["'\w\d]$/.test(trimmed)) {
          result.push(line + ')'.repeat(diff));
          continue;
        }
      }
      result.push(line);
    }
    return result.join('\n');
  }

  // 4. Rewrite file reads to use demo data
  function rewriteFileReads(code) {
    const hasFileRead = /pd\.(read_excel|read_csv|read_table)\s*\(/.test(code);
    if (!hasFileRead) return { code, injected: false };

    // Replace the assignment: `something = pd.read_*(...)` → `something = _demo_data.copy()`
    let rewritten = code.replace(
      /pd\.(read_excel|read_csv|read_table)\s*\([^)]*\)/g,
      '_demo_data.copy()'
    );
    return { code: DEMO_DATA_PY + '\n' + rewritten, injected: true };
  }

  // ── Parse imports ─────────────────────────────────────────────────────────
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

  async function ensureMicropipPackages(imports, outputEl) {
    for (const imp of imports) {
      if (PYODIDE_BUILTIN.has(imp) || installedPackages.has(imp)) continue;
      const pkg = MICROPIP_MAP[imp];
      if (!pkg) continue;
      outputEl.textContent = `⏳ Installing ${pkg}…`;
      try {
        await pyodide.runPythonAsync(`import micropip\nawait micropip.install('${pkg}')`);
        installedPackages.add(imp);
      } catch (e) {
        console.warn(`micropip install ${pkg} failed:`, e.message);
      }
    }
  }

  function nsHas(name) {
    try { return pyodide.runPython(`'${name}' in globals()`); }
    catch(e) { return false; }
  }

  // Build preamble: alias guards + missing `data` injection
  function buildPreamble(code) {
    const lines = [];
    for (const { alias, pattern, inject } of ALIAS_GUARDS) {
      if (!pattern.test(code)) continue;
      const inCell = new RegExp(`(import\\s+\\S+\\s+as\\s+${alias}\\b|import\\s+${alias}\\b)`).test(code);
      if (inCell || nsHas(alias)) continue;
      lines.push(inject);
    }
    // Auto-inject `data` if used but not defined and not in namespace
    const usesData = /\bdata\b/.test(code) && !/\bdata\s*=/.test(code) && !nsHas('data');
    if (usesData) lines.push(DEMO_DATA_PY);
    return lines.join('\n');
  }

  // ── Run code ──────────────────────────────────────────────────────────────
  async function runCode(rawCode, outputEl, bannerEl, noteEl) {
    outputEl.className = 'pyodide-output has-content is-loading';
    outputEl.textContent = '⏳ Loading Python runtime…';
    bannerEl.style.display = 'none';
    if (noteEl) noteEl.style.display = 'none';

    // Apply all sanitizers in order
    let code = dedent(rawCode);
    code = removeOuterReturns(code);
    code = fixUnmatchedParens(code);

    const imports = parseImports(code);
    const incompatible = findIncompatible(imports);
    if (incompatible.length) {
      outputEl.className = 'pyodide-output';
      outputEl.textContent = '';
      bannerEl.style.display = 'flex';
      bannerEl.innerHTML = `<span class="banner-icon">⚠️</span><div><strong>Browser-incompatible library:</strong> ${incompatible.map(l=>`<code>${l}</code>`).join(', ')} cannot run in the browser. Run locally with <code>pip install ${incompatible.join(' ')}</code>.</div>`;
      return;
    }

    outputEl.textContent = '⏳ Starting Python…';
    await ensureMicropipPackages(imports, outputEl);

    // Rewrite file reads
    const { code: fileCode, injected: fileInjected } = rewriteFileReads(code);
    code = fileCode;

    // Alias + data preamble
    const preamble = buildPreamble(code);
    const fullCode = preamble ? preamble + '\n' + code : code;
    const preambleLines = preamble ? preamble.split('\n').length : 0;

    if (noteEl && fileInjected) {
      noteEl.textContent = '📊 Using demo dataset (real file not available in browser)';
      noteEl.style.display = 'block';
    }

    await pyodide.runPythonAsync(`_capture.truncate(0); _capture.seek(0); _plot_images.clear()`);

    let textOut = '', isError = false, plotImages = [];

    try {
      const ret = await pyodide.runPythonAsync(fullCode, { globals: sharedNamespace });

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
      if (preambleLines > 0) {
        msg = msg.replace(/line (\d+)/g, (m, n) => {
          const adj = parseInt(n) - preambleLines;
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

    const note = document.createElement('div');
    note.className = 'pyodide-note';
    note.style.display = 'none';

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
        whenReady(() => runCode(textarea.value, output, banner, note).then(after).catch(after));
        return;
      }
      runCode(textarea.value, output, banner, note).then(after).catch(after);
    });

    wrap.appendChild(toolbar);
    wrap.appendChild(textarea);
    wrap.appendChild(note);
    wrap.appendChild(banner);
    wrap.appendChild(output);
    return wrap;
  }

  // ── Extract plain text ────────────────────────────────────────────────────
  function extractCode(el) {
    const pre = el.querySelector('pre');
    if (!pre) return '';
    return pre.textContent.replace(/^\n/, '').replace(/\n$/, '');
  }

  // ── Transform cells ───────────────────────────────────────────────────────
  function transformCells() {
    injectStyles();

    // 1. Jupyter notebook cells (div.cell_input)
    document.querySelectorAll('div.cell_input').forEach(cellInput => {
      const highlight = cellInput.querySelector('.highlight-python, .highlight-ipython3, .highlight-default');
      if (!highlight) return;
      const code = extractCode(highlight);
      if (!code.trim()) return;
      highlight.parentNode.replaceChild(buildEditor(code), highlight);
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

    loadPyodideIfNeeded();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', transformCells);
  } else {
    transformCells();
  }
})();
