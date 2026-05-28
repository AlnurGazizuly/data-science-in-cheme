/**
 * Pyodide Inline Runner v10
 * All sanitizers run before execution to handle common notebook issues.
 */

(function () {
  'use strict';

  const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js';
  const LOAD_PACKAGES = ['numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn'];

  const MICROPIP_MAP = {
    'sklearn':'scikit-learn','cv2':'opencv-python','PIL':'Pillow',
    'missingno':'missingno','shap':'shap','skopt':'scikit-optimize',
    'imblearn':'imbalanced-learn','xgboost':'xgboost','plotly':'plotly',
    'statsmodels':'statsmodels','nltk':'nltk','seaborn':'seaborn','joblib':'joblib',
  };
  const PYODIDE_BUILTIN = new Set(['numpy','pandas','matplotlib','scipy','sklearn','scikit-learn','sqlite3','ssl','hashlib','hmac','lzma','bz2','zlib']);
  const BROWSER_INCOMPATIBLE = new Set(['streamlit','torch','gradio','langchain','tensorflow','keras','flask','django','fastapi','celery','redis','psycopg2']);
  const ALIAS_GUARDS = [
    { alias:'np',  pattern:/\bnp\./,  inject:'import numpy as np' },
    { alias:'pd',  pattern:/\bpd\./,  inject:'import pandas as pd' },
    { alias:'plt', pattern:/\bplt\./, inject:'import matplotlib.pyplot as plt' },
    { alias:'sns', pattern:/\bsns\./, inject:'import seaborn as sns' },
    { alias:'sp',  pattern:/\bsp\./,  inject:'import scipy as sp' },
  ];

  // Demo data: ALL numeric columns so corr/PCA/MinMaxScaler/quantile always work.
  // NaN values scattered so missing-value heatmaps show real patterns.
  // Feature1/2/3 have realistic (non-perfect) correlations.
  // Values column has an outlier (120) so IQR outlier detection works.
  const DEMO_DATA_PY = `
import pandas as pd
import numpy as np
_demo_data = pd.DataFrame({
    'Temperature':   [295, 305, np.nan, 298, 310, 320, np.nan, 302, 318, 308],
    'Pressure':      [1.1, 1.8, 2.5, np.nan, 2.1, 2.9, 0.9, np.nan, 2.7, 2.2],
    'FlowRate':      [10.2, 15.4, 20.1, 12.3, np.nan, 22.5, 9.8, 14.6, np.nan, 18.9],
    'Concentration': [0.5, 1.0, 1.5, 0.7, 1.2, np.nan, 0.4, 0.9, 1.6, 1.3],
    'Feature1':      [2.3, 4.1, 3.8, 5.2, 1.9, 6.7, 3.3, 4.8, 2.1, 5.5],
    'Feature2':      [1.1, 3.9, 2.7, 4.5, 2.2, 5.1, 1.8, 3.6, 2.9, 4.2],
    'Feature3':      [8.2, 6.4, 9.1, 5.7, 7.8, 4.3, 8.9, 6.1, 7.3, 5.2],
    'Feature4':      [3.1, 5.5, 4.2, 6.8, 2.5, 7.3, 3.9, 5.1, 2.8, 6.2],
    'sepal_length':  [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
    'sepal_width':   [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
    'petal_length':  [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
    'petal_width':   [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
    'Values':        [10, 12, 15, 18, 19, 120, 14, 13, 16, 17],
    'Category':      [1, 2, 1, 3, 2, 1, 3, 2, 1, 2],
    'target':        [2.1, 3.8, 5.2, 4.7, 6.3, 7.1, 3.5, 4.2, 5.8, 6.9],
    'MSP':           [45.2, 62.1, 78.5, 55.3, 89.4, 103.2, 51.7, 67.8, 82.1, 94.6],
})
data = _demo_data.copy()
df = _demo_data.copy()
`;

  let pyodide = null, pyodideLoading = false, pyodideReady = false;
  const pendingRuns = [];
  let sharedNamespace = null;
  const installedPackages = new Set();

  const CSS = `
.pyodide-editor-wrap{margin:.5rem 0 .25rem;border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,.12);background:#1e1e2e;font-family:'Fira Mono','Consolas',monospace}
.pyodide-toolbar{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:rgba(0,0,0,.25);border-bottom:1px solid rgba(255,255,255,.08)}
.pyodide-toolbar .py-label{font-size:.7rem;color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:.08em;user-select:none}
.pyodide-run-btn{display:inline-flex;align-items:center;gap:5px;padding:3px 12px;border:none;border-radius:4px;background:#4f9cf9;color:#fff;font-size:.78rem;font-weight:600;cursor:pointer;transition:background .15s;user-select:none}
.pyodide-run-btn:hover:not(:disabled){background:#3a88e8}
.pyodide-run-btn:disabled{background:#555;cursor:wait}
.pyodide-run-btn svg{width:11px;height:11px;fill:#fff}
.pyodide-textarea{display:block;width:100%;box-sizing:border-box;min-height:52px;padding:10px 14px;background:transparent;color:#cdd6f4;font-family:inherit;font-size:.88rem;line-height:1.55;border:none;outline:none;resize:vertical;tab-size:4;white-space:pre;overflow-x:auto}
.pyodide-output{display:none;border-top:1px solid rgba(255,255,255,.08);padding:8px 14px;min-height:1.5em;font-family:inherit;font-size:.85rem;white-space:pre-wrap;word-break:break-word;color:#a6e3a1;background:rgba(0,0,0,.18);line-height:1.5}
.pyodide-output.has-content{display:block}
.pyodide-output.is-error{color:#f38ba8}
.pyodide-output.is-loading{color:rgba(255,255,255,.45);font-style:italic}
.pyodide-output img{max-width:100%;margin-top:6px;border-radius:4px;display:block}
.pyodide-note{padding:4px 14px;font-size:.75rem;color:rgba(255,255,255,.35);font-style:italic;background:rgba(0,0,0,.12);border-top:1px solid rgba(255,255,255,.05)}
.pyodide-banner{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;background:rgba(243,139,168,.08);border-top:1px solid rgba(243,139,168,.2);color:#f5c2e7;font-size:.82rem;line-height:1.5}
.pyodide-banner .banner-icon{font-size:1.1rem;flex-shrink:0}
.pyodide-banner code{background:rgba(255,255,255,.1);border-radius:3px;padding:1px 5px;font-family:monospace}
.pyodide-textarea:focus{background:rgba(255,255,255,.03)}
`;

  function injectStyles() {
    if (document.getElementById('pyodide-runner-css')) return;
    const s = document.createElement('style');
    s.id = 'pyodide-runner-css';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

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
      } catch(e) { pyodideLoading = false; console.error('Pyodide load failed:', e); }
    };
    script.onerror = () => { pyodideLoading = false; };
    document.head.appendChild(script);
  }

  function whenReady(fn) { if (pyodideReady) fn(); else pendingRuns.push(fn); }

  // ── Sanitizer 1: Smart dedent ─────────────────────────────────────────────
  function dedent(code) {
    const lines = code.split('\n');
    const nonEmpty = lines.filter(l => l.trim().length > 0);
    if (!nonEmpty.length) return code;
    const minIndent = Math.min(...nonEmpty.map(l => l.match(/^(\s*)/)[1].length));
    if (minIndent > 0) return lines.map(l => l.slice(minIndent)).join('\n');
    // minIndent=0: first line at col 0, but rest may have spurious indent
    const rest = nonEmpty.slice(1);
    if (rest.length === 0) return code;
    const restMin = Math.min(...rest.map(l => l.match(/^(\s*)/)[1].length));
    if (restMin > 0 && !nonEmpty[0].trimEnd().endsWith(':')) {
      return lines.map(l => {
        if (l.trim().length === 0) return l;
        const ind = l.match(/^(\s*)/)[1].length;
        return ind >= restMin ? l.slice(restMin) : l;
      }).join('\n');
    }
    return code;
  }

  // ── Sanitizer 2: Remove bare `return` outside functions ───────────────────
  function removeOuterReturns(code) {
    let depth = 0;
    return code.split('\n').map(line => {
      const stripped = line.trim();
      if (/^(def |class |async def )\w/.test(stripped)) depth++;
      if (depth > 0 && stripped.length > 0 && !stripped.startsWith('#')) {
        const lead = line.match(/^(\s*)/)[1].length;
        if (lead === 0 && !/^(def |class |async def |else:|elif |except|finally)/.test(stripped)) depth = 0;
      }
      if (depth === 0 && /^\s*return(\s|$)/.test(line))
        return line.replace(/(\s*)return(\s*)/, '$1# return$2');
      return line;
    }).join('\n');
  }

  // ── Sanitizer 3: Fix unmatched open parens ────────────────────────────────
  function fixUnmatchedParens(code) {
    return code.split('\n').map(line => {
      if (line.trim().startsWith('#')) return line;
      const diff = (line.match(/\(/g)||[]).length - (line.match(/\)/g)||[]).length;
      if (diff > 0 && !/[,(\[{\\]$/.test(line.trimEnd()) && line.trimEnd().length > 0)
        return line + ')'.repeat(diff);
      return line;
    }).join('\n');
  }

  // ── Sanitizer 4: Rewrite file reads to demo data ──────────────────────────
  function rewriteFileReads(code) {
    if (!/pd\.(read_excel|read_csv|read_table)\s*\(/.test(code)) return { code, injected: false };
    const rewritten = code.replace(/pd\.(read_excel|read_csv|read_table)\s*\([^)]*\)/g, '_demo_data.copy()');
    return { code: DEMO_DATA_PY + '\n' + rewritten, injected: true };
  }

  // ── Sanitizer 5: Fix ndarray → list for list-only methods ────────────────
  function fixNdarrayListMethods(code) {
    const pattern = /\b(\w+)\.(append|remove|insert|pop)\s*\(/g;
    const varNames = new Set();
    let m;
    while ((m = pattern.exec(code)) !== null) varNames.add(m[1]);
    const guards = [];
    for (const v of varNames) {
      if (new RegExp(`\\b${v}\\s*=\\s*\\[`).test(code)) continue;
      guards.push(`if '${v}' in globals() and hasattr(globals()['${v}'], 'tolist'): ${v} = list(${v})`);
    }
    return guards.length ? guards.join('\n') + '\n' + code : code;
  }

  // ── Sanitizer 6: Fix placeholder column names ─────────────────────────────
  function fixPlaceholderColumns(code) {
    return code
      .replace(/['"]ColumnName['"]/g, "'Temperature'")
      .replace(/['"]column_name['"]/g, "'Temperature'");
  }

  // ── Sanitizer 6b: Fix .hist() layout that's too small for actual columns ────
  // Pattern: df.hist(..., layout=(r, c), ...) where r*c < num_columns
  // Replace with auto-computed layout via Python math.ceil
  function fixHistLayout(code) {
    if (!/\.hist\s*\(/.test(code)) return code;
    // Replace any layout=(N, M) inside a .hist() call with a safe computed version
    // We inject a Python helper that computes the right layout
    let c = code;
    // Replace layout=(digits, digits) with a sentinel; inject helper
    if (/layout\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)/.test(c)) {
      c = c.replace(/layout\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)/g, (match, r, c2) => {
        // Keep original layout but add import math guard before the hist call
        return match; // we'll handle it via Python preamble instead
      });
      // Prepend a Python snippet that will compute a safe layout at runtime
      const guard = `import math as _math
_hist_ncols = 3
`;
      c = guard + c;
      // Replace the layout=(r,c) with a dynamically computed one
      c = c.replace(/layout\s*=\s*\(\s*\d+\s*,\s*\d+\s*\)/g,
        'layout=(_math.ceil(len([x for x in _hist_df.columns if _hist_df[x].dtype != object]) / _hist_ncols), _hist_ncols)');
      // Capture the df variable used in .hist()
      const dfHist = /\b(df_normalized|df_num|df|data)\s*\.hist/.exec(c);
      if (dfHist) {
        c = c.replace('_hist_df', dfHist[1]).replace('_hist_df', dfHist[1]);
        // Replace all remaining _hist_df with actual df var
        c = c.replace(/_hist_df/g, dfHist[1]);
      }
    }
    return c;
  }

  // ── Sanitizer 7: Fix numeric-only operations on mixed-type DataFrames ─────
  // Handles: .corr(), .quantile(), MinMaxScaler, StandardScaler, PCA
  // All fail when DataFrame has string/object columns.
  // Strategy: inject a _num_df helper and use it wherever numeric-only ops occur.
  function fixNumericOps(code) {
    let c = code;

    // .corr() on df/data → .select_dtypes(include='number').corr()
    c = c.replace(/\b(df|data)\.(corr)\s*\(/g, "$1.select_dtypes(include='number').$2(");

    // .quantile() on df/data → .select_dtypes(include='number').quantile()
    c = c.replace(/\b(df|data)\.(quantile)\s*\(/g, "$1.select_dtypes(include='number').$2(");

    // For MinMaxScaler / StandardScaler / PCA applied to the whole df:
    // Pattern: scaler.fit_transform(df, ...) or scaler.fit_transform(data, ...)
    // Replace with a version that uses numeric-only columns
    // Also handles pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    if (/(MinMaxScaler|StandardScaler|PCA)/.test(c)) {
      // Find variable name (df or data)
      const dfVar = /\bdf\b/.test(c) ? 'df' : 'data';
      // Prepend numeric df extraction
      const numVar = `${dfVar}_num`;
      const guard = `${numVar} = ${dfVar}.select_dtypes(include='number').fillna(${dfVar}.select_dtypes(include='number').median())\n_pca_ncomp = min(len(${numVar}.columns), len(${numVar}) - 1)\n`;
      // Replace fit_transform(df) → fit_transform(df_num)
      c = c.replace(new RegExp(`\\.fit_transform\\s*\\(\\s*${dfVar}\\s*\\)`, 'g'), `.fit_transform(${numVar})`);
      // Replace fit_transform(df, columns=df.columns) → fit_transform(df_num, columns=df_num.columns)
      c = c.replace(
        new RegExp(`\\.fit_transform\\s*\\(\\s*${dfVar}\\s*,\\s*columns\\s*=\\s*${dfVar}\\.columns\\s*\\)`, 'g'),
        `.fit_transform(${numVar}, columns=${numVar}.columns)`
      );
      // Replace pd.DataFrame(..., columns=df.columns) → pd.DataFrame(..., columns=df_num.columns)
      c = c.replace(
        new RegExp(`columns\\s*=\\s*${dfVar}\\.columns`, 'g'),
        `columns=${numVar}.columns`
      );
      // Replace PCA(n_components=len(df.columns)) → PCA(n_components=len(df_num.columns))
      c = c.replace(
        new RegExp(`len\\(${dfVar}\\.columns\\)`, 'g'),
        `_pca_ncomp`
      );
      // Replace df.columns[:-1] style references in correlation checks
      c = c.replace(
        new RegExp(`${dfVar}\\.columns\\[`, 'g'),
        `${numVar}.columns[`
      );
      // Replace correlation_matrix = df.corr() (already handled above but catch df_num.corr case)
      // Also fix: for col in correlation_matrix.columns: → works as-is after corr() patch
      // Cap any remaining literal n_components referencing columns length
      c = c.replace(/n_components\s*=\s*len\s*\([^)]+\.columns\)/g, 'n_components=_pca_ncomp');
      c = guard + c;
    }

    // IQR alignment fix: when outliers_condition compares df (mixed) against Q1/Q3 (numeric Series)
    // The Q1/Q3 come from df.select_dtypes(...).quantile() → they're indexed on numeric cols only
    // But df < Q1 tries to compare all columns → alignment error
    // Fix: replace (df < (Q1 with (df.select_dtypes(include='number') < (Q1
    c = c.replace(/\(\s*(df|data)\s*(<|>|<=|>=)\s*\(/g, "($1.select_dtypes(include='number') $2 (");
    // Also: df[~outliers_condition.any(axis=1)] works if outliers_condition is on numeric df

    return c;
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
      } catch(e) { console.warn(`micropip: ${pkg} failed`, e.message); }
    }
  }

  function nsHas(name) {
    try { return pyodide.runPython(`'${name}' in globals()`); } catch(e) { return false; }
  }

  function buildPreamble(code) {
    const lines = [];
    for (const { alias, pattern, inject } of ALIAS_GUARDS) {
      if (!pattern.test(code)) continue;
      const inCell = new RegExp(`(import\\s+\\S+\\s+as\\s+${alias}\\b|import\\s+${alias}\\b)`).test(code);
      if (inCell || nsHas(alias)) continue;
      lines.push(inject);
    }
    const usesData = /\bdata\b/.test(code) && !/\bdata\s*=/.test(code) && !nsHas('data');
    const usesDf   = /\bdf\b/.test(code)   && !/\bdf\s*=/.test(code)   && !nsHas('df');
    if (usesData || usesDf) {
      lines.push(DEMO_DATA_PY);
      // If sklearn is involved, auto-provide clean (NaN-free) versions
      const usesSklearn = /from\s+sklearn|import\s+sklearn|train_test_split|LinearRegression|RandomForest|cross_val/.test(code);
      if (usesSklearn) {
        lines.push(`# Auto-clean for sklearn: drop NaN rows from demo data\n_df_clean = df.dropna().reset_index(drop=True)\n_data_clean = data.dropna().reset_index(drop=True)\ndf = _df_clean\ndata = _data_clean\n`);
      }
    }
    return lines.join('\n');
  }

  // ── Run code ──────────────────────────────────────────────────────────────
  // Fix sklearn API compatibility issues (e.g. squared param removed in newer sklearn)
  function fixSklearnCompat(code) {
    // make_scorer with squared= param removed in sklearn 1.4+
    // Remove squared=True (it was the default — plain MSE)
    code = code.replace(/make_scorer\s*\(\s*mean_squared_error\s*,([^)]*),\s*squared\s*=\s*True([^)]*)\)/g,
      'make_scorer(mean_squared_error,$1$2)');
    code = code.replace(/make_scorer\s*\(\s*mean_squared_error\s*,\s*squared\s*=\s*True\s*\)/g,
      'make_scorer(mean_squared_error, greater_is_better=False)');
    // Remove squared=False (was sqrt(MSE) = RMSE) — replace whole make_scorer with RMSE scorer
    code = code.replace(/make_scorer\s*\(\s*mean_squared_error\s*,([^)]*),\s*squared\s*=\s*False([^)]*)\)/g,
      'make_scorer(lambda y,p: __import__("numpy").sqrt(mean_squared_error(y,p)), greater_is_better=False)');
    // Direct mean_squared_error(y,p, squared=True/False) calls
    code = code.replace(/mean_squared_error\s*\(([^)]*),\s*squared\s*=\s*True([^)]*)\)/g,
      'mean_squared_error($1$2)');
    code = code.replace(/mean_squared_error\s*\(([^)]*),\s*squared\s*=\s*False([^)]*)\)/g,
      '__import__("numpy").sqrt(mean_squared_error($1$2))');
    return code;
  }

  async function runCode(rawCode, outputEl, bannerEl, noteEl) {
    outputEl.className = 'pyodide-output has-content is-loading';
    outputEl.textContent = '⏳ Loading Python runtime…';
    bannerEl.style.display = 'none';
    if (noteEl) noteEl.style.display = 'none';

    let code = dedent(rawCode);
    code = fixHistLayout(code);
    code = removeOuterReturns(code);
    code = fixUnmatchedParens(code);
    code = fixNdarrayListMethods(code);
    code = fixPlaceholderColumns(code);
    code = fixNumericOps(code);
    code = fixSklearnCompat(code);

    const imports = parseImports(code);
    const incompatible = findIncompatible(imports);
    if (incompatible.length) {
      outputEl.className = 'pyodide-output';
      outputEl.textContent = '';
      bannerEl.style.display = 'flex';
      bannerEl.innerHTML = `<span class="banner-icon">⚠️</span><div><strong>Browser-incompatible:</strong> ${incompatible.map(l=>`<code>${l}</code>`).join(', ')} can't run in browser. Use <code>pip install ${incompatible.join(' ')}</code> locally.</div>`;
      return;
    }

    outputEl.textContent = '⏳ Starting Python…';
    await ensureMicropipPackages(imports, outputEl);

    const { code: fileCode, injected: fileInjected } = rewriteFileReads(code);
    code = fileCode;

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
    } catch(err) {
      isError = true;
      const cap = pyodide.runPython(`_capture.getvalue()`);
      let msg = (cap||'') + (cap?'\n':'') + err.message;
      msg = msg.replace(/File "\/lib\/python[^"]*",\s*/g,'').replace(/\n\s*\^{5,}\n/g,'\n');
      if (preambleLines > 0) {
        msg = msg.replace(/line (\d+)/g, (_,n) => {
          const adj = parseInt(n) - preambleLines;
          return adj > 0 ? `line ${adj}` : `line ${n}`;
        });
      }
      textOut = msg;
    }

    outputEl.className = 'pyodide-output has-content' + (isError ? ' is-error' : '');
    outputEl.textContent = '';
    if (textOut.trim()) outputEl.textContent = textOut;
    else if (!plotImages.length && !isError) outputEl.textContent = '✓ Done (no output)';
    for (const b64 of plotImages) {
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + b64;
      img.style.maxWidth = '100%';
      outputEl.appendChild(img);
    }
  }

  // ── Build editor ──────────────────────────────────────────────────────────
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
    textarea.rows = Math.max(2, (originalCode.match(/\n/g)||[]).length + 1);
    textarea.spellcheck = false;
    textarea.autocomplete = textarea.autocorrect = textarea.autocapitalize = 'off';
    textarea.addEventListener('input', () => {
      textarea.rows = Math.max(2, (textarea.value.match(/\n/g)||[]).length + 1);
    });
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const s = textarea.selectionStart;
        textarea.value = textarea.value.slice(0,s) + '    ' + textarea.value.slice(textarea.selectionEnd);
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

  function extractCode(el) {
    const pre = el.querySelector('pre');
    if (!pre) return '';
    return pre.textContent.replace(/^\n/,'').replace(/\n$/,'');
  }

  function transformCells() {
    injectStyles();
    document.querySelectorAll('div.cell_input').forEach(cellInput => {
      const h = cellInput.querySelector('.highlight-python,.highlight-ipython3,.highlight-default');
      if (!h) return;
      const code = extractCode(h);
      if (!code.trim()) return;
      h.parentNode.replaceChild(buildEditor(code), h);
    });
    const sel = ['div.highlight-python:not(.cell_input *)','div.highlight-ipython3:not(.cell_input *)','div.highlight-default:not(.cell_input *)','div.highlight-shell:not(.cell_input *)'].join(',');
    document.querySelectorAll(sel).forEach(el => {
      if (el.closest('.pyodide-editor-wrap')) return;
      const outer = el.closest('.highlight-python,.highlight-ipython3,.highlight-default,.highlight-shell') || el;
      const code = extractCode(outer);
      if (!code.trim()) return;
      outer.parentNode.replaceChild(buildEditor(code), outer);
    });
    loadPyodideIfNeeded();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', transformCells);
  else transformCells();
})();
