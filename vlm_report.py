import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from vlm_types import BenchmarkResult, DatasetImage


def generate_html_gallery(
    items: list[DatasetImage],
    results_by_model: dict[str, BenchmarkResult],
    output_html_path: Path,
    split: str,
):
    """Generate self-contained interactive HTML report comparing model predictions and feedback."""
    model_keys = list(results_by_model.keys())

    # Build lookup by rel_path for each model
    lookups = {
        k: {r.rel_path: r for r in results_by_model[k].records} for k in model_keys
    }

    def to_b64(path: Path) -> str:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((360, 360))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=80)
                return (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buf.getvalue()).decode()
                )
        except Exception:
            return ""

    cards_html = []
    for item in items:
        rel = item.rel_path
        gt_cls = "proper" if item.gt_proper else "improper"
        gt_badge = (
            '<span class="badge badge-proper">GT: Proper</span>'
            if item.gt_proper
            else '<span class="badge badge-improper">GT: Improper</span>'
        )

        b64_src = to_b64(item.path)
        model_rows = []
        any_misclassified = False

        for k in model_keys:
            rec = lookups[k].get(rel)
            if not rec:
                continue

            if not rec.is_correct:
                any_misclassified = True

            pred_badge = (
                f'<span class="badge badge-correct">✓ {rec.pred_label.capitalize()}</span>'
                if rec.is_correct
                else f'<span class="badge badge-wrong">✗ {rec.pred_label.capitalize()}</span>'
            )

            feedback_text = (
                rec.feedback.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            model_rows.append(
                f"""
                <div class="model-row" data-model="{k}" data-correct="{str(rec.is_correct).lower()}">
                    <div class="model-header">
                        <span class="model-title">{results_by_model[k].display_name}</span>
                        <div class="model-badges">
                            {pred_badge}
                            <span class="latency-tag">{rec.latency_ms:.0f} ms</span>
                        </div>
                    </div>
                    <div class="feedback-bubble">"{feedback_text}"</div>
                </div>
                """
            )

        err_class = "card-has-error" if any_misclassified else "card-all-correct"
        card_markup = f"""
        <div class="gallery-card {err_class}" data-gt="{gt_cls}" data-err="{str(any_misclassified).lower()}" data-search="{rel.lower()}">
            <div class="card-img-container">
                <img src="{b64_src}" alt="{rel}" loading="lazy"/>
                <div class="img-overlay">{gt_badge}</div>
            </div>
            <div class="card-body">
                <div class="card-filename" title="{rel}">{Path(rel).name}</div>
                <div class="models-container">
                    {"".join(model_rows)}
                </div>
            </div>
        </div>
        """
        cards_html.append(card_markup)

    # Default sort by Macro F1 descending (with accuracy as tie-breaker)
    sorted_model_keys = sorted(
        model_keys,
        key=lambda k: (results_by_model[k].macro_f1, results_by_model[k].accuracy),
        reverse=True,
    )

    # Month abbreviation lookup for chronological sorting
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }

    summary_rows = []
    for k in sorted_model_keys:
        res = results_by_model[k]
        date_str = res.release_date
        # Convert "Feb 2026" to "2026-02" for accurate chronological sorting
        parts = date_str.split()
        if len(parts) == 2 and parts[0] in month_map:
            sort_val = f"{parts[1]}-{month_map[parts[0]]}"
        else:
            sort_val = date_str

        summary_rows.append(
            f"""
            <tr class="model-summary-row" data-model="{k}" onclick="selectModelFromTable('{k}')" title="Click row to filter feedback gallery below">
                <td data-val="{res.display_name}">
                    <div class="model-name-cell">
                        <a href="https://huggingface.co/{res.hf_id}" target="_blank" rel="noopener noreferrer" class="model-hf-link" onclick="event.stopPropagation()" title="Open {res.display_name} on Hugging Face">
                            <strong>{res.display_name}</strong>
                            <span class="hf-link-icon">↗</span>
                        </a>
                        <span class="filter-hint">🔍 filter</span>
                    </div>
                </td>
                <td data-val="{sort_val}"><span class="year-pill">{date_str}</span></td>
                <td data-val="{res.accuracy:.4f}">{res.accuracy:.1f}%</td>
                <td data-val="{res.macro_f1:.4f}">{res.macro_f1:.1f}%</td>
                <td data-val="{res.parse_rate:.4f}">{res.parse_rate:.1f}%</td>
                <td data-val="{res.mean_latency_ms:.4f}">{res.mean_latency_ms:.1f} ms</td>
            </tr>
            """
        )

    model_options = [
        f'<option value="{k}">{results_by_model[k].display_name}</option>'
        for k in sorted_model_keys
    ]

    split_title = split.upper()
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM Scooter Parking Gallery - {split_title} Split</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --proper: #10b981;
            --improper: #ef4444;
            --primary: #3b82f6;
            --accent: #8b5cf6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ margin-bottom: 24px; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        h1 {{ font-size: 28px; font-weight: 700; color: #fff; }}
        .split-pill {{
            background: var(--primary);
            color: #fff;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        p.subtitle {{ color: var(--text-muted); font-size: 15px; margin-bottom: 20px; }}
        
        .metrics-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .metrics-card h2 {{ font-size: 18px; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th, td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); font-weight: 600; background: rgba(255,255,255,0.02); }}
        th.sortable {{
            cursor: pointer;
            user-select: none;
            transition: background 0.15s, color 0.15s;
            position: relative;
            white-space: nowrap;
        }}
        th.sortable:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
        }}
        th.sortable .sort-icon {{
            display: inline-block;
            margin-left: 6px;
            font-size: 11px;
            opacity: 0.45;
            vertical-align: middle;
        }}
        th.sortable.sorted-asc,
        th.sortable.sorted-desc {{
            color: #fff;
            background: rgba(59, 130, 246, 0.12);
        }}
        th.sortable.sorted-asc .sort-icon,
        th.sortable.sorted-desc .sort-icon {{
            opacity: 1;
            color: var(--primary);
        }}

        tr.model-summary-row {{
            cursor: pointer;
            transition: background-color 0.15s;
        }}
        tr.model-summary-row:hover {{
            background: rgba(59, 130, 246, 0.08);
        }}
        tr.model-summary-row.selected-model-row {{
            background: rgba(59, 130, 246, 0.22) !important;
        }}
        .model-name-cell {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .model-hf-link {{
            color: #fff;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: color 0.15s;
        }}
        .model-hf-link:hover {{
            color: var(--primary);
            text-decoration: underline;
        }}
        .hf-link-icon {{
            font-size: 11px;
            opacity: 0.55;
            transition: opacity 0.15s, transform 0.15s;
        }}
        .model-hf-link:hover .hf-link-icon {{
            opacity: 1;
            transform: translate(1px, -1px);
        }}
        .filter-hint {{
            font-size: 11px;
            color: var(--primary);
            opacity: 0;
            transition: opacity 0.15s;
            background: rgba(59, 130, 246, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
        }}
        tr.model-summary-row:hover .filter-hint {{
            opacity: 1;
        }}
        .table-hint {{
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        .year-pill {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.06);
            color: #cbd5e1;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            background: var(--surface);
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        .controls-left {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }}
        .controls-right {{
            display: flex;
            align-items: center;
        }}
        .control-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-label {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .filter-select {{
            background: #0f172a;
            border: 1px solid var(--border);
            color: #fff;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            outline: none;
            transition: border-color 0.15s;
            max-width: 280px;
        }}
        .filter-select:focus {{
            border-color: var(--primary);
        }}
        .btn-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .btn {{
            background: #273549;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.15s ease;
        }}
        .btn:hover, .btn.active {{
            background: var(--primary);
            border-color: var(--primary);
            color: #fff;
        }}
        .search-box {{
            background: #0f172a;
            border: 1px solid var(--border);
            color: #fff;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 13px;
            width: 260px;
        }}
        .search-box:focus {{ outline: none; border-color: var(--primary); }}
        
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        .gallery-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .gallery-card:hover {{
            transform: translateY(-2px);
            border-color: #475569;
        }}
        .gallery-card.model-error {{
            border-color: rgba(239, 68, 68, 0.55);
        }}
        .gallery-card.model-correct {{
            border-color: rgba(16, 185, 129, 0.3);
        }}
        .card-img-container {{
            position: relative;
            background: #090d16;
            height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .card-img-container img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .img-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-proper {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }}
        .badge-improper {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }}
        .badge-correct {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-wrong {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .latency-tag {{ font-size: 11px; color: var(--text-muted); font-family: monospace; }}
        
        .card-body {{ padding: 14px; flex-grow: 1; display: flex; flex-direction: column; }}
        .card-filename {{
            font-size: 12px;
            color: var(--text-muted);
            font-family: monospace;
            margin-bottom: 12px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .models-container {{ display: flex; flex-direction: column; gap: 10px; }}
        .model-row {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .model-title {{ font-size: 13px; font-weight: 600; color: #cbd5e1; }}
        .model-badges {{ display: flex; gap: 6px; align-items: center; }}
        .feedback-bubble {{
            font-size: 12.5px;
            color: #e2e8f0;
            font-style: italic;
            line-height: 1.4;
            background: rgba(255,255,255,0.03);
            padding: 6px 10px;
            border-radius: 6px;
            border-left: 3px solid var(--primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <h1>🛴 Scooter Parking VLM Benchmark</h1>
                <span class="split-pill">{split_title} SET</span>
            </div>
            <p class="subtitle">Zero-shot evaluation on classification accuracy and natural language actionable feedback generation ({len(items)} images).</p>
        </header>

        <div class="metrics-card">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
                <h2>Summary Performance Table ({split_title})</h2>
                <span class="table-hint">💡 Click model name for Hugging Face page &bull; Click row to filter gallery &bull; Click column to sort</span>
            </div>
            <table id="summaryTable">
                <thead>
                    <tr>
                        <th class="sortable" onclick="sortTable(0, 'string')">Model <span class="sort-icon">⇅</span></th>
                        <th class="sortable" onclick="sortTable(1, 'string')">Release Date <span class="sort-icon">⇅</span></th>
                        <th class="sortable" onclick="sortTable(2, 'number')">Accuracy <span class="sort-icon">⇅</span></th>
                        <th class="sortable sorted-desc" onclick="sortTable(3, 'number')">Macro F1 <span class="sort-icon">▼</span></th>
                        <th class="sortable" onclick="sortTable(4, 'number')">Valid JSON Rate <span class="sort-icon">⇅</span></th>
                        <th class="sortable" onclick="sortTable(5, 'number')">Mean Latency <span class="sort-icon">⇅</span></th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(summary_rows)}
                </tbody>
            </table>
        </div>

        <div class="controls" id="galleryControls">
            <div class="controls-left">
                <div class="control-item">
                    <label for="modelFilterSelect" class="control-label">Model:</label>
                    <select id="modelFilterSelect" class="filter-select" onchange="onModelFilterChange(this.value)">
                        <option value="all">All Models (Combined)</option>
                        {"".join(model_options)}
                    </select>
                </div>
                <div class="btn-group">
                    <button class="btn active" id="btnFilterAll" onclick="filterGallery('all')">Show All ({len(items)})</button>
                    <button class="btn" id="btnFilterErrors" onclick="filterGallery('errors')">Misclassified Only</button>
                    <button class="btn" id="btnFilterProper" onclick="filterGallery('proper')">GT: Proper</button>
                    <button class="btn" id="btnFilterImproper" onclick="filterGallery('improper')">GT: Improper</button>
                </div>
            </div>
            <div class="controls-right">
                <input type="text" id="searchBox" class="search-box" placeholder="Filter by image name or keyword..." oninput="searchGallery()"/>
            </div>
        </div>

        <div class="gallery-grid" id="galleryGrid">
            {"".join(cards_html)}
        </div>
    </div>

    <script>
        let currentSortCol = 3;
        let currentSortDir = 'desc';

        function sortTable(colIndex, type) {{
            const table = document.getElementById('summaryTable');
            if (!table) return;
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const ths = table.querySelectorAll('thead th');

            if (currentSortCol === colIndex) {{
                currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
            }} else {{
                currentSortCol = colIndex;
                if (colIndex === 0 || colIndex === 5) {{
                    currentSortDir = 'asc';
                }} else {{
                    currentSortDir = 'desc';
                }}
            }}

            rows.sort((a, b) => {{
                const cellA = a.children[colIndex];
                const cellB = b.children[colIndex];
                let valA = cellA.getAttribute('data-val') !== null ? cellA.getAttribute('data-val') : cellA.innerText.trim();
                let valB = cellB.getAttribute('data-val') !== null ? cellB.getAttribute('data-val') : cellB.innerText.trim();

                if (type === 'number') {{
                    valA = parseFloat(valA) || 0;
                    valB = parseFloat(valB) || 0;
                    return currentSortDir === 'asc' ? valA - valB : valB - valA;
                }} else {{
                    valA = valA.toLowerCase();
                    valB = valB.toLowerCase();
                    if (valA < valB) return currentSortDir === 'asc' ? -1 : 1;
                    if (valA > valB) return currentSortDir === 'asc' ? 1 : -1;
                    return 0;
                }}
            }});

            ths.forEach((th, idx) => {{
                th.classList.remove('sorted-asc', 'sorted-desc');
                const icon = th.querySelector('.sort-icon');
                if (idx === colIndex) {{
                    th.classList.add(currentSortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
                    if (icon) icon.textContent = currentSortDir === 'asc' ? '▲' : '▼';
                }} else {{
                    if (icon) icon.textContent = '⇅';
                }}
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}

        let currentFilter = 'all';
        let currentSelectedModel = 'all';

        function onModelFilterChange(modelKey) {{
            currentSelectedModel = modelKey;
            highlightSelectedModelRow(modelKey);
            updateFilterCounts();
            applyFilters();
        }}

        function selectModelFromTable(modelKey) {{
            const select = document.getElementById('modelFilterSelect');
            if (select) {{
                if (currentSelectedModel === modelKey) {{
                    select.value = 'all';
                    onModelFilterChange('all');
                }} else {{
                    select.value = modelKey;
                    onModelFilterChange(modelKey);
                    const controls = document.getElementById('galleryControls');
                    if (controls) {{
                        controls.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                }}
            }}
        }}

        function highlightSelectedModelRow(modelKey) {{
            document.querySelectorAll('tr.model-summary-row').forEach(row => {{
                if (row.getAttribute('data-model') === modelKey && modelKey !== 'all') {{
                    row.classList.add('selected-model-row');
                }} else {{
                    row.classList.remove('selected-model-row');
                }}
            }});
        }}

        function updateFilterCounts() {{
            const cards = document.querySelectorAll('.gallery-card');
            let countAll = 0;
            let countErrors = 0;
            let countProper = 0;
            let countImproper = 0;

            cards.forEach(card => {{
                const gt = card.getAttribute('data-gt');
                const modelRows = Array.from(card.querySelectorAll('.model-row'));

                if (currentSelectedModel === 'all') {{
                    countAll++;
                    if (card.getAttribute('data-err') === 'true') countErrors++;
                    if (gt === 'proper') countProper++;
                    if (gt === 'improper') countImproper++;
                }} else {{
                    const row = modelRows.find(r => r.getAttribute('data-model') === currentSelectedModel);
                    if (row) {{
                        countAll++;
                        if (row.getAttribute('data-correct') === 'false') countErrors++;
                        if (gt === 'proper') countProper++;
                        if (gt === 'improper') countImproper++;
                    }}
                }}
            }});

            const btnAll = document.getElementById('btnFilterAll');
            const btnErrors = document.getElementById('btnFilterErrors');
            const btnProper = document.getElementById('btnFilterProper');
            const btnImproper = document.getElementById('btnFilterImproper');

            if (btnAll) btnAll.textContent = `Show All (${{countAll}})`;
            if (btnErrors) {{
                if (currentSelectedModel === 'all') {{
                    btnErrors.textContent = `Misclassified (Any Model) (${{countErrors}})`;
                }} else {{
                    btnErrors.textContent = `Misclassified Only (${{countErrors}})`;
                }}
            }}
            if (btnProper) btnProper.textContent = `GT: Proper (${{countProper}})`;
            if (btnImproper) btnImproper.textContent = `GT: Improper (${{countImproper}})`;
        }}

        function filterGallery(type) {{
            currentFilter = type;
            document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            applyFilters();
        }}

        function searchGallery() {{
            applyFilters();
        }}

        function applyFilters() {{
            const query = document.getElementById('searchBox').value.toLowerCase().trim();
            const cards = document.querySelectorAll('.gallery-card');

            cards.forEach(card => {{
                const gt = card.getAttribute('data-gt');
                const modelRows = Array.from(card.querySelectorAll('.model-row'));
                let cardHasModel = false;
                let isMisclassified = false;

                if (currentSelectedModel === 'all') {{
                    cardHasModel = true;
                    isMisclassified = card.getAttribute('data-err') === 'true';
                    modelRows.forEach(r => {{ r.style.display = ''; }});
                    card.classList.remove('model-error', 'model-correct');
                }} else {{
                    const row = modelRows.find(r => r.getAttribute('data-model') === currentSelectedModel);
                    if (row) {{
                        cardHasModel = true;
                        isMisclassified = row.getAttribute('data-correct') === 'false';
                        modelRows.forEach(r => {{
                            r.style.display = (r === row) ? '' : 'none';
                        }});
                        card.classList.toggle('model-error', isMisclassified);
                        card.classList.toggle('model-correct', !isMisclassified);
                    }} else {{
                        cardHasModel = false;
                    }}
                }}

                if (!cardHasModel) {{
                    card.style.display = 'none';
                    return;
                }}

                let matchesFilter = true;
                if (currentFilter === 'errors') matchesFilter = isMisclassified;
                else if (currentFilter === 'proper') matchesFilter = (gt === 'proper');
                else if (currentFilter === 'improper') matchesFilter = (gt === 'improper');

                const rel = card.getAttribute('data-search') || '';
                let visibleText = rel;
                modelRows.forEach(r => {{
                    if (r.style.display !== 'none') {{
                        visibleText += ' ' + r.innerText.toLowerCase();
                    }}
                }});

                let matchesSearch = query === '' || visibleText.includes(query);

                if (matchesFilter && matchesSearch) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        // Initialize counts on load
        updateFilterCounts();
    </script>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML Gallery written to {output_html_path}")


def save_benchmark_reports(
    items: list[DatasetImage],
    current_results: dict[str, BenchmarkResult],
    out_dir: Path,
    split: str,
):
    """Save raw JSON predictions and split-specific interactive HTML gallery."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save split-specific predictions JSON
    pred_json_path = out_dir / f"vlm_predictions_{split}.json"
    pred_json_data = {k: v.model_dump() for k, v in current_results.items()}
    pred_json_path.write_text(json.dumps(pred_json_data, indent=2), encoding="utf-8")
    print(f"Saved {split} raw predictions to {pred_json_path}")

    # 2. Save split-specific HTML gallery
    html_path = out_dir / f"vlm_results_{split}.html"
    generate_html_gallery(items, current_results, html_path, split=split)
