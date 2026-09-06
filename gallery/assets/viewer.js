// Client-side VLM Benchmark Gallery viewer
const monthMap = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
};

let currentData = null;
let currentSortCol = 3;
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

async function loadBenchmarkGallery(split) {
    const tableBody = document.getElementById('summaryTableBody');
    const galleryGrid = document.getElementById('galleryGrid');
    const modelSelect = document.getElementById('modelFilterSelect');

    try {
        const response = await fetch(`../vlm_output/vlm_predictions_${split}.json`);
        if (!response.ok) {
            throw new Error(`Failed to load predictions: HTTP ${response.status}`);
        }
        currentData = await response.json();
        renderGallery(split, currentData);
    } catch (err) {
        console.error('Error loading benchmark data:', err);
        if (galleryGrid) {
            galleryGrid.innerHTML = `
                <div class="state-message" style="grid-column: 1/-1;">
                    <p style="color: #f87171; font-weight: 600; margin-bottom: 8px;">Failed to load benchmark predictions for ${split} split.</p>
                    <p style="font-size: 13px; color: var(--text-muted);">Could not locate <code>vlm_predictions_${split}.json</code>.</p>
                </div>
            `;
        }
    }
}

function renderGallery(split, resultsByModel) {
    const modelKeys = Object.keys(resultsByModel);
    if (modelKeys.length === 0) {
        document.getElementById('galleryGrid').innerHTML = `
            <div class="state-message" style="grid-column: 1/-1;">
                No models evaluated yet for ${split} split.
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
                <td data-val="${escapeHtml(sortVal)}"><span class="year-pill">${escapeHtml(dateStr)}</span></td>
                <td data-val="${res.accuracy.toFixed(4)}">${res.accuracy.toFixed(1)}%</td>
                <td data-val="${res.macro_f1.toFixed(4)}">${res.macro_f1.toFixed(1)}%</td>
                <td data-val="${res.parse_rate.toFixed(4)}">${res.parse_rate.toFixed(1)}%</td>
                <td data-val="${res.mean_latency_ms.toFixed(4)}">${res.mean_latency_ms.toFixed(1)} ms</td>
            </tr>
        `;
    }).join('');

    // 2. Render Model Dropdown
    const modelSelect = document.getElementById('modelFilterSelect');
    modelSelect.innerHTML = '<option value="all">All Models (Combined)</option>' +
        sortedModelKeys.map(k => `<option value="${escapeHtml(k)}">${escapeHtml(resultsByModel[k].display_name)}</option>`).join('');

    // 3. Build unique images list from first model's records
    const recordsFirst = resultsByModel[modelKeys[0]].records;
    const lookups = {};
    modelKeys.forEach(k => {
        lookups[k] = {};
        resultsByModel[k].records.forEach(r => {
            lookups[k][r.rel_path] = r;
        });
    });

    // Update total count button
    const btnAll = document.getElementById('btnFilterAll');
    if (btnAll) btnAll.textContent = `Show All (${recordsFirst.length})`;

    const galleryGrid = document.getElementById('galleryGrid');
    galleryGrid.innerHTML = recordsFirst.map(item => {
        const rel = item.rel_path;
        const gtProper = item.gt_proper;
        const gtCls = gtProper ? 'proper' : 'improper';
        const gtBadge = gtProper 
            ? '<span class="badge badge-proper">GT: Proper</span>' 
            : '<span class="badge badge-improper">GT: Improper</span>';

        let anyMisclassified = false;
        const modelRowsHtml = modelKeys.map(k => {
            const rec = lookups[k][rel];
            if (!rec) return '';
            if (!rec.is_correct) anyMisclassified = true;

            const predBadge = rec.is_correct
                ? `<span class="badge badge-correct">✓ ${escapeHtml(rec.pred_label)}</span>`
                : `<span class="badge badge-wrong">✗ ${escapeHtml(rec.pred_label)}</span>`;

            return `
                <div class="model-row" data-model="${escapeHtml(k)}" data-correct="${rec.is_correct}">
                    <div class="model-header">
                        <span class="model-title">${escapeHtml(resultsByModel[k].display_name)}</span>
                        <div class="model-badges">
                            ${predBadge}
                            <span class="latency-tag">${rec.latency_ms.toFixed(0)} ms</span>
                        </div>
                    </div>
                    <div class="feedback-bubble">"${escapeHtml(rec.feedback)}"</div>
                </div>
            `;
        }).join('');

        const errClass = anyMisclassified ? 'card-has-error' : 'card-all-correct';
        const fileName = rel.split('/').pop().split('\\').pop();

        // Optimized thumbnail located in gallery/images/<split>/<fileName>
        const imgSrc = `images/${split}/${escapeHtml(fileName)}`;

        return `
            <div class="gallery-card ${errClass}" data-gt="${gtCls}" data-err="${anyMisclassified}" data-search="${escapeHtml(rel.toLowerCase())}">
                <div class="card-img-container">
                    <img src="${imgSrc}" alt="${escapeHtml(rel)}" loading="lazy"/>
                    <div class="img-overlay">${gtBadge}</div>
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
        if (colIndex === 0 || colIndex === 5) {
            currentSortDir = 'asc';
        } else {
            currentSortDir = 'desc';
        }
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
    document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
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

        let matchesSearch = query === '' || visibleText.includes(query);

        if (matchesFilter && matchesSearch) {
            card.style.display = 'flex';
        } else {
            card.style.display = 'none';
        }
    });
}
