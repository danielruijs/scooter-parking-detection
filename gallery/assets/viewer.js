// Client-side VLM Benchmark Gallery viewer
const monthMap = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
};


function parseParamCount(str) {
    if (!str) return 0;
    const s = str.trim().toUpperCase();
    const num = parseFloat(s);
    if (isNaN(num)) return 0;
    if (s.endsWith('B')) return num * 1000;
    if (s.endsWith('M')) return num;
    return num;
}

function getQuantClass(quant) {
    if (quant === '4-bit') return 'quant-4bit';
    if (quant === 'FP16') return 'quant-fp16';
    return '';
}

const datasets = { val: null, train: null, combined: null };
let activeSplit = 'combined';
let currentSortCol = 5;
let currentSortDir = 'desc';
let currentFilter = 'all';
let currentSelectedModel = 'all';

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function computeMetrics(records, baseInfo) {
    const total = records.length;
    if (total === 0) {
        return { ...baseInfo, accuracy: 0, macro_f1: 0, parse_rate: 0, mean_latency_ms: 0, records: [] };
    }

    const yTrue = records.map(r => r.gt_proper ? 1 : 0);
    const yPred = records.map(r => r.pred_proper === true ? 1 : (r.pred_proper === false ? 0 : -1));

    let correctCount = 0;
    let parseCount = 0;
    let latencySum = 0;

    for (let i = 0; i < total; i++) {
        if (yTrue[i] === yPred[i]) correctCount++;
        if (records[i].parse_success) parseCount++;
        latencySum += records[i].latency_ms || 0;
    }

    // Macro F1 matching scikit-learn average='macro', zero_division=0
    const classes = Array.from(new Set([...yTrue, ...yPred])).sort((a, b) => a - b);
    let f1Sum = 0;
    for (const c of classes) {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < total; i++) {
            if (yTrue[i] === c && yPred[i] === c) tp++;
            if (yTrue[i] !== c && yPred[i] === c) fp++;
            if (yTrue[i] === c && yPred[i] !== c) fn++;
        }
        const p = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const r = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1 = (p + r) > 0 ? (2 * p * r) / (p + r) : 0;
        f1Sum += f1;
    }
    const macroF1 = classes.length > 0 ? (f1Sum / classes.length) * 100 : 0;

    return {
        ...baseInfo,
        accuracy: (correctCount / total) * 100,
        macro_f1: macroF1,
        parse_rate: (parseCount / total) * 100,
        mean_latency_ms: latencySum / total,
        records
    };
}

let isInitialized = false;

async function initGallery() {
    if (isInitialized) return;
    isInitialized = true;

    const params = new URLSearchParams(window.location.search);
    const splitParam = params.get('split');
    if (['combined', 'val', 'train'].includes(splitParam)) {
        activeSplit = splitParam;
    }

    try {
        const [valData, trainData] = await Promise.all([
            fetchData('val'),
            fetchData('train')
        ]);

        datasets.val = valData;
        datasets.train = trainData;

        // Build combined dataset
        const combined = {};
        const allModelKeys = Array.from(new Set([...Object.keys(valData || {}), ...Object.keys(trainData || {})]));
        for (const k of allModelKeys) {
            const valEntry = valData?.[k];
            const trainEntry = trainData?.[k];
            const base = valEntry || trainEntry;
            const records = [...(valEntry?.records || []), ...(trainEntry?.records || [])];
            combined[k] = computeMetrics(records, base);
        }
        datasets.combined = combined;

        switchSplit(activeSplit);
    } catch (err) {
        console.error('Error loading benchmark data:', err);
        const galleryGrid = document.getElementById('galleryGrid');
        if (galleryGrid) {
            galleryGrid.innerHTML = `
                <div class="state-message" style="grid-column: 1/-1;">
                    <p style="color: #f87171; font-weight: 600; margin-bottom: 8px;">Failed to load benchmark predictions.</p>
                    <p style="font-size: 13px; color: var(--text-muted);">${escapeHtml(err.message)}</p>
                </div>
            `;
        }
    }
}

async function fetchData(split) {
    let response = await fetch(`./vlm_output/vlm_predictions_${split}.json`);
    if (!response.ok) {
        response = await fetch(`../vlm_output/vlm_predictions_${split}.json`);
    }
    if (!response.ok) {
        throw new Error(`HTTP ${response.status} loading predictions for ${split}`);
    }
    return await response.json();
}

function switchSplit(split) {
    if (!datasets[split]) return;
    activeSplit = split;

    // Update split buttons
    document.querySelectorAll('.split-btn').forEach(b => b.classList.remove('active'));
    const btnMap = { combined: 'splitBtnCombined', val: 'splitBtnVal', train: 'splitBtnTrain' };
    document.getElementById(btnMap[split])?.classList.add('active');

    // Update heading
    const titleMap = {
        combined: 'Summary Performance Table (Combined: Train + Val)',
        val: 'Summary Performance Table (Validation Split)',
        train: 'Summary Performance Table (Train Split)'
    };
    const headingEl = document.getElementById('tableHeading');
    if (headingEl) headingEl.textContent = titleMap[split];

    // Sync URL param
    const url = new URL(window.location);
    url.searchParams.set('split', split);
    window.history.replaceState({}, '', url);

    renderGallery(datasets[split]);
}

function renderGallery(resultsByModel) {
    const modelKeys = Object.keys(resultsByModel);
    const galleryGrid = document.getElementById('galleryGrid');
    if (modelKeys.length === 0) {
        galleryGrid.innerHTML = `
            <div class="state-message" style="grid-column: 1/-1;">
                No models evaluated yet for this split.
            </div>
        `;
        return;
    }

    // 1. Render Summary Table
    const sortedModelKeys = [...modelKeys].sort((a, b) => {
        const resA = resultsByModel[a];
        const resB = resultsByModel[b];
        if (resB.macro_f1 !== resA.macro_f1) return resB.macro_f1 - resA.macro_f1;
        return resB.accuracy - resA.accuracy;
    });

    const tbody = document.getElementById('summaryTableBody');
    tbody.innerHTML = sortedModelKeys.map(k => {
        const res = resultsByModel[k];
        const dateStr = res.release_date || '';
        const parts = dateStr.split(' ');
        const sortVal = (parts.length === 2 && monthMap[parts[0]]) 
            ? `${parts[1]}-${monthMap[parts[0]]}` 
            : dateStr;

        const paramsStr = res.params || '';
        const quantStr = res.quantization || '';
        const paramSortVal = parseParamCount(paramsStr);
        const quantClass = getQuantClass(quantStr);

        return `
            <tr class="model-summary-row" data-model="${escapeHtml(k)}" onclick="selectModelFromTable('${escapeHtml(k)}')" title="Click row to filter feedback gallery below">
                <td data-val="${escapeHtml(res.display_name)}">
                    <div class="model-name-cell">
                        <a href="https://huggingface.co/${escapeHtml(res.hf_id)}" target="_blank" rel="noopener noreferrer" class="model-hf-link" onclick="event.stopPropagation()" title="Open ${escapeHtml(res.display_name)} on Hugging Face">
                            <strong>${escapeHtml(res.display_name)}</strong>
                            <span class="hf-link-icon">↗</span>
                        </a>
                        <span class="filter-hint">🔍 filter</span>
                    </div>
                </td>
                <td data-val="${paramSortVal}"><span class="param-pill">${escapeHtml(paramsStr)}</span></td>
                <td data-val="${escapeHtml(quantStr)}"><span class="quant-pill ${quantClass}">${escapeHtml(quantStr)}</span></td>
                <td data-val="${escapeHtml(sortVal)}"><span class="year-pill">${escapeHtml(dateStr)}</span></td>
                <td data-val="${res.accuracy.toFixed(4)}">${res.accuracy.toFixed(1)}%</td>
                <td data-val="${res.macro_f1.toFixed(4)}">${res.macro_f1.toFixed(1)}%</td>
                <td data-val="${res.parse_rate.toFixed(4)}">${res.parse_rate.toFixed(1)}%</td>
                <td data-val="${res.mean_latency_ms.toFixed(4)}">${res.mean_latency_ms.toFixed(1)} ms</td>
            </tr>
        `;
    }).join('');

    // 2. Render Model Dropdown (preserve selection if exists)
    const modelSelect = document.getElementById('modelFilterSelect');
    const prevSelected = currentSelectedModel;
    modelSelect.innerHTML = '<option value="all">All Models (Combined)</option>' +
        sortedModelKeys.map(k => {
            const item = resultsByModel[k];
            const q = item.quantization || '';
            const label = q ? `${item.display_name} • ${q}` : item.display_name;
            return `<option value="${escapeHtml(k)}">${escapeHtml(label)}</option>`;
        }).join('');

    if (sortedModelKeys.includes(prevSelected)) {
        modelSelect.value = prevSelected;
        currentSelectedModel = prevSelected;
        highlightSelectedModelRow(prevSelected);
    } else {
        modelSelect.value = 'all';
        currentSelectedModel = 'all';
        highlightSelectedModelRow('all');
    }

    // 3. Build unique images list from first model's records
    const recordsFirst = resultsByModel[modelKeys[0]].records;
    const lookups = {};
    modelKeys.forEach(k => {
        lookups[k] = {};
        resultsByModel[k].records.forEach(r => {
            lookups[k][r.rel_path] = r;
        });
    });

    const btnAll = document.getElementById('btnFilterAll');
    if (btnAll) btnAll.textContent = `Show All (${recordsFirst.length})`;

    galleryGrid.innerHTML = recordsFirst.map(item => {
        const rel = item.rel_path;
        const gtProper = item.gt_proper;
        const gtCls = gtProper ? 'proper' : 'improper';
        const gtBadge = gtProper 
            ? '<span class="badge badge-proper">GT: Proper</span>' 
            : '<span class="badge badge-improper">GT: Improper</span>';

        const itemSplit = rel.includes('/val/') || rel.includes('\\val\\') ? 'val' : 'train';
        const splitBadge = activeSplit === 'combined'
            ? `<span class="badge badge-split-${itemSplit}">${itemSplit.toUpperCase()}</span>`
            : '';

        let anyMisclassified = false;
        const modelRowsHtml = modelKeys.map(k => {
            const rec = lookups[k][rel];
            if (!rec) return '';
            if (!rec.is_correct) anyMisclassified = true;

            const predBadge = rec.is_correct
                ? `<span class="badge badge-correct">✓ ${escapeHtml(rec.pred_label)}</span>`
                : `<span class="badge badge-wrong">✗ ${escapeHtml(rec.pred_label)}</span>`;

            const item = resultsByModel[k];
            const q = item.quantization || '';
            const subtag = q ? `<span class="model-subtag">${escapeHtml(q)}</span>` : '';

            return `
                <div class="model-row" data-model="${escapeHtml(k)}" data-correct="${rec.is_correct}">
                    <div class="model-header">
                        <span class="model-title">${escapeHtml(item.display_name)} ${subtag}</span>
                        <div class="model-badges">
                            ${predBadge}
                            <span class="latency-tag" title="Inference latency measured on NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM)">${rec.latency_ms.toFixed(0)} ms</span>
                        </div>
                    </div>
                    <div class="feedback-bubble">"${escapeHtml(rec.feedback)}"</div>
                </div>
            `;
        }).join('');

        const errClass = anyMisclassified ? 'card-has-error' : 'card-all-correct';
        const fileName = rel.split('/').pop().split('\\').pop();
        const imgSrc = `images/${itemSplit}/${escapeHtml(fileName)}`;

        return `
            <div class="gallery-card ${errClass}" data-gt="${gtCls}" data-err="${anyMisclassified}" data-search="${escapeHtml(rel.toLowerCase())}">
                <div class="card-img-container">
                    <img src="${imgSrc}" alt="${escapeHtml(rel)}" loading="lazy"/>
                    <div class="img-overlay">${gtBadge}${splitBadge}</div>
                </div>
                <div class="card-body">
                    <div class="card-filename" title="${escapeHtml(rel)}">${escapeHtml(fileName)}</div>
                    <div class="models-container">
                        ${modelRowsHtml}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    updateFilterCounts();
    applyFilters();
}

function sortTable(colIndex, type) {
    const table = document.getElementById('summaryTable');
    if (!table) return;
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const ths = table.querySelectorAll('thead th');

    if (currentSortCol === colIndex) {
        currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
    } else {
        currentSortCol = colIndex;
        // Ascending by default: Model (0), Quantization (2), Latency (7)
        // Descending by default: Params (1), Release Date (3), Accuracy (4), Macro F1 (5), Valid JSON (6)
        currentSortDir = (colIndex === 0 || colIndex === 2 || colIndex === 7) ? 'asc' : 'desc';
    }

    rows.sort((a, b) => {
        const cellA = a.children[colIndex];
        const cellB = b.children[colIndex];
        let valA = cellA.getAttribute('data-val') !== null ? cellA.getAttribute('data-val') : cellA.innerText.trim();
        let valB = cellB.getAttribute('data-val') !== null ? cellB.getAttribute('data-val') : cellB.innerText.trim();

        if (type === 'number') {
            valA = parseFloat(valA) || 0;
            valB = parseFloat(valB) || 0;
            return currentSortDir === 'asc' ? valA - valB : valB - valA;
        } else {
            valA = valA.toLowerCase();
            valB = valB.toLowerCase();
            if (valA < valB) return currentSortDir === 'asc' ? -1 : 1;
            if (valA > valB) return currentSortDir === 'asc' ? 1 : -1;
            return 0;
        }
    });

    ths.forEach((th, idx) => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        const icon = th.querySelector('.sort-icon');
        if (idx === colIndex) {
            th.classList.add(currentSortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            if (icon) icon.textContent = currentSortDir === 'asc' ? '▲' : '▼';
        } else {
            if (icon) icon.textContent = '⇅';
        }
    });

    rows.forEach(row => tbody.appendChild(row));
}

function onModelFilterChange(modelKey) {
    currentSelectedModel = modelKey;
    highlightSelectedModelRow(modelKey);
    updateFilterCounts();
    applyFilters();
}

function selectModelFromTable(modelKey) {
    const select = document.getElementById('modelFilterSelect');
    if (select) {
        if (currentSelectedModel === modelKey) {
            select.value = 'all';
            onModelFilterChange('all');
        } else {
            select.value = modelKey;
            onModelFilterChange(modelKey);
            const controls = document.getElementById('galleryControls');
            if (controls) {
                controls.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    }
}

function highlightSelectedModelRow(modelKey) {
    const rows = document.querySelectorAll('tr.model-summary-row');
    rows.forEach(r => {
        if (modelKey !== 'all' && r.getAttribute('data-model') === modelKey) {
            r.classList.add('selected-model-row');
        } else {
            r.classList.remove('selected-model-row');
        }
    });
}

function updateFilterCounts() {
    const cards = document.querySelectorAll('.gallery-card');
    let errCount = 0;
    cards.forEach(card => {
        if (currentSelectedModel === 'all') {
            if (card.getAttribute('data-err') === 'true') errCount++;
        } else {
            const mRow = card.querySelector(`.model-row[data-model="${currentSelectedModel}"]`);
            if (mRow && mRow.getAttribute('data-correct') === 'false') {
                errCount++;
            }
        }
    });

    const errBtn = document.getElementById('btnFilterErrors');
    if (errBtn) {
        const modelLabel = currentSelectedModel === 'all' ? '' : ' for Model';
        errBtn.textContent = `Misclassified Only (${errCount}${modelLabel})`;
    }
}

function filterGallery(type) {
    currentFilter = type;
    document.querySelectorAll('.btn-group .btn').forEach(b => {
        if (['btnFilterAll', 'btnFilterErrors', 'btnFilterProper', 'btnFilterImproper'].includes(b.id)) {
            b.classList.remove('active');
        }
    });
    const activeMap = {
        'all': 'btnFilterAll',
        'errors': 'btnFilterErrors',
        'proper': 'btnFilterProper',
        'improper': 'btnFilterImproper'
    };
    if (activeMap[type]) {
        const btn = document.getElementById(activeMap[type]);
        if (btn) btn.classList.add('active');
    }
    applyFilters();
}

function searchGallery() {
    applyFilters();
}

function applyFilters() {
    const cards = document.querySelectorAll('.gallery-card');
    const searchBox = document.getElementById('searchBox');
    const query = searchBox ? searchBox.value.toLowerCase().trim() : '';

    cards.forEach(card => {
        const gt = card.getAttribute('data-gt');
        const anyErr = card.getAttribute('data-err') === 'true';
        let cardMatchesModelError = false;

        const modelRows = card.querySelectorAll('.model-row');
        modelRows.forEach(row => {
            const rModel = row.getAttribute('data-model');
            const isCorrect = row.getAttribute('data-correct') === 'true';
            if (currentSelectedModel === 'all' || rModel === currentSelectedModel) {
                row.style.display = 'block';
                if (!isCorrect) cardMatchesModelError = true;
            } else {
                row.style.display = 'none';
            }
        });

        card.classList.remove('model-error', 'model-correct');
        if (currentSelectedModel !== 'all') {
            if (cardMatchesModelError) {
                card.classList.add('model-error');
            } else {
                card.classList.add('model-correct');
            }
        }

        let matchesFilter = false;
        if (currentFilter === 'all') {
            matchesFilter = true;
        } else if (currentFilter === 'errors') {
            matchesFilter = currentSelectedModel === 'all' ? anyErr : cardMatchesModelError;
        } else if (currentFilter === 'proper') {
            matchesFilter = (gt === 'proper');
        } else if (currentFilter === 'improper') {
            matchesFilter = (gt === 'improper');
        }

        let visibleText = card.getAttribute('data-search') || '';
        modelRows.forEach(r => {
            if (r.style.display !== 'none') {
                visibleText += ' ' + r.innerText.toLowerCase();
            }
        });

        const matchesSearch = query === '' || visibleText.includes(query);

        if (matchesFilter && matchesSearch) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}

// Global window bindings
window.initGallery = initGallery;
window.loadBenchmarkGallery = initGallery;
window.switchSplit = switchSplit;
window.sortTable = sortTable;
window.onModelFilterChange = onModelFilterChange;
window.selectModelFromTable = selectModelFromTable;
window.filterGallery = filterGallery;
window.searchGallery = searchGallery;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => initGallery());
} else {
    initGallery();
}

