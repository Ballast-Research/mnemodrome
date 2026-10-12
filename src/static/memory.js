/**
 * Mnemodrome Admin UI - Memory Browser shared helpers (Feature 039)
 *
 * Depends on app.js (apiGet, apiPost, escapeHtml, formatTimestamp,
 * formatNumber, formatBytes, showToast, handleApiError, setActiveNavLink).
 * No external libraries; works offline.
 *
 * Safety: every value from the API is passed through escapeHtml() before it
 * is placed in markup. The only server HTML is the markdown body, which the
 * server renders with raw HTML disabled.
 */

const MEMORY_API = '/admin/api/memory';

/** Read a query-string parameter. */
function qp(name, fallback = '') {
    const v = new URLSearchParams(window.location.search).get(name);
    return v === null ? fallback : v;
}

/** Replace the URL query string without reloading (keeps links shareable). */
function setQuery(params) {
    const url = new URL(window.location.href);
    Object.entries(params).forEach(([k, v]) => {
        if (v === '' || v === null || v === undefined) url.searchParams.delete(k);
        else url.searchParams.set(k, v);
    });
    window.history.replaceState({}, '', url.toString());
}

/** Badge for fact validity status. */
function statusBadge(status) {
    const map = { valid: 'success', invalidated: 'warning', superseded: 'info' };
    return `<span class="badge badge-${map[status] || 'info'}">${escapeHtml(status || 'unknown')}</span>`;
}

/** Badge for backend health. */
function healthBadge(status) {
    const ok = status === 'healthy';
    const cls = ok ? 'success' : (status === 'disabled' || status === 'not_initialized') ? 'warning' : 'error';
    return `<span class="badge badge-${cls}">${escapeHtml(status || 'unknown')}</span>`;
}

function shortDate(iso) {
    if (!iso) return '--';
    try { return formatTimestamp(iso); } catch (e) { return String(iso); }
}

/**
 * Render a pager into `container`. Calls onPage(offset) on click.
 * Uses offset/limit/total from a server response.
 */
function renderPager(container, { offset, limit, total }, onPage) {
    const el = typeof container === 'string' ? document.getElementById(container) : container;
    if (!el) return;
    const pages = Math.max(1, Math.ceil(total / limit));
    const current = Math.floor(offset / limit) + 1;
    const first = total === 0 ? 0 : offset + 1;
    const last = Math.min(total, offset + limit);
    el.innerHTML = `
        <div class="memory-pager">
            <span class="text-muted">Showing ${formatNumber(first)}–${formatNumber(last)} of ${formatNumber(total)}</span>
            <div class="btn-group" style="margin-top:0;">
                <button type="button" class="btn btn-secondary" ${current <= 1 ? 'disabled' : ''} data-go="0">« First</button>
                <button type="button" class="btn btn-secondary" ${current <= 1 ? 'disabled' : ''} data-go="${Math.max(0, offset - limit)}">‹ Prev</button>
                <span class="memory-pager-page">Page ${formatNumber(current)} / ${formatNumber(pages)}</span>
                <button type="button" class="btn btn-secondary" ${current >= pages ? 'disabled' : ''} data-go="${offset + limit}">Next ›</button>
                <button type="button" class="btn btn-secondary" ${current >= pages ? 'disabled' : ''} data-go="${(pages - 1) * limit}">Last »</button>
            </div>
        </div>`;
    el.querySelectorAll('button[data-go]').forEach(b => {
        b.addEventListener('click', () => onPage(parseInt(b.dataset.go, 10)));
    });
}

/** Simple key/value table for header blocks and detail panels. */
function kvTable(obj, keys = null) {
    const entries = keys ? keys.filter(k => obj[k] !== undefined).map(k => [k, obj[k]]) : Object.entries(obj);
    if (!entries.length) return '<p class="text-muted">None</p>';
    return `<table class="memory-kv"><tbody>${entries.map(([k, v]) => `
        <tr><th>${escapeHtml(k)}</th><td>${v === null || v === undefined || v === '' ? '<span class="text-muted">—</span>' : escapeHtml(String(v))}</td></tr>`).join('')}</tbody></table>`;
}

/** Open the slide-in detail panel with the given (already escaped) HTML. */
function openPanel(title, bodyHtml) {
    let panel = document.getElementById('memory-panel');
    if (!panel) {
        panel = document.createElement('aside');
        panel.id = 'memory-panel';
        panel.className = 'memory-panel';
        const header = document.createElement('div');
        header.className = 'memory-panel-header';
        const h2 = document.createElement('h2');
        h2.id = 'memory-panel-title';
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'btn btn-secondary';
        close.setAttribute('aria-label', 'Close');
        close.textContent = '×';
        close.addEventListener('click', closePanel);
        header.appendChild(h2);
        header.appendChild(close);
        const body = document.createElement('div');
        body.className = 'memory-panel-body';
        body.id = 'memory-panel-body';
        panel.appendChild(header);
        panel.appendChild(body);
        document.body.appendChild(panel);
        document.addEventListener('keydown', e => { if (e.key === 'Escape') closePanel(); });
    }
    document.getElementById('memory-panel-title').textContent = title;
    document.getElementById('memory-panel-body').innerHTML = bodyHtml;
    panel.classList.add('open');
}

function closePanel() {
    const panel = document.getElementById('memory-panel');
    if (panel) panel.classList.remove('open');
}

function spinnerHtml() {
    return '<div class="spinner-container"><div class="spinner"></div></div>';
}

function emptyHtml(msg) {
    return `<div style="padding:2rem;text-align:center;"><p class="text-muted">${escapeHtml(msg)}</p></div>`;
}

/** Fetch and render one markdown file into the panel. */
async function showFile(type, name) {
    openPanel(name, spinnerHtml());
    try {
        const f = await apiGet(`${MEMORY_API}/files/${encodeURIComponent(type)}/${encodeURIComponent(name)}`);
        openPanel(f.title || name, `
            <div class="memory-file-meta">
                ${kvTable({ file: f.path, modified: shortDate(f.modified), size: formatBytes(f.size) })}
                ${Object.keys(f.header || {}).length ? `<h3>Header</h3>${kvTable(f.header)}` : ''}
                ${f.truncated ? '<p class="badge badge-warning">File truncated to 2 MB</p>' : ''}
            </div>
            <div class="memory-markdown">${f.html}</div>`);
    } catch (e) {
        openPanel(name, `<p class="badge badge-error">${escapeHtml(e.message)}</p>`);
    }
}

/**
 * Draw a small radial graph as inline SVG: one centre entity, neighbours on a ring.
 * rels: [{other_name, other_uuid, relation, direction, status}]
 */
function renderEntityGraph(container, centreName, rels, onNodeClick) {
    const el = typeof container === 'string' ? document.getElementById(container) : container;
    if (!el) return;
    const MAX = 40;
    const shown = rels.slice(0, MAX);
    const size = 560, cx = size / 2, cy = size / 2;
    const radius = shown.length <= 8 ? 170 : 220;
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.setAttribute('class', 'memory-graph');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `Relationships of ${centreName}`);

    const defs = document.createElementNS(ns, 'defs');
    const marker = document.createElementNS(ns, 'marker');
    marker.setAttribute('id', 'arrow');
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '7');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('orient', 'auto-start-reverse');
    const head = document.createElementNS(ns, 'path');
    head.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    head.setAttribute('class', 'memory-graph-arrow');
    marker.appendChild(head);
    defs.appendChild(marker);
    svg.appendChild(defs);

    const nodes = [];
    shown.forEach((r, i) => {
        const angle = (2 * Math.PI * i) / shown.length - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        nodes.push({ x, y, r });

        const line = document.createElementNS(ns, 'line');
        const from = r.direction === 'out' ? { x: cx, y: cy } : { x, y };
        const to = r.direction === 'out' ? { x, y } : { x: cx, y: cy };
        const dx = to.x - from.x, dy = to.y - from.y, len = Math.hypot(dx, dy) || 1;
        const pad = 26;
        line.setAttribute('x1', from.x + (dx / len) * pad);
        line.setAttribute('y1', from.y + (dy / len) * pad);
        line.setAttribute('x2', to.x - (dx / len) * pad);
        line.setAttribute('y2', to.y - (dy / len) * pad);
        line.setAttribute('class', `memory-graph-edge status-${r.status || 'valid'}`);
        line.setAttribute('marker-end', 'url(#arrow)');
        svg.appendChild(line);

        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', (cx + x) / 2);
        label.setAttribute('y', (cy + y) / 2 - 4);
        label.setAttribute('class', 'memory-graph-edge-label');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = r.relation || '';
        svg.appendChild(label);
    });

    const addNode = (x, y, name, uuid, centre) => {
        const g = document.createElementNS(ns, 'g');
        g.setAttribute('class', `memory-graph-node${centre ? ' centre' : ''}`);
        if (uuid && onNodeClick) {
            g.style.cursor = 'pointer';
            g.addEventListener('click', () => onNodeClick(uuid));
        }
        const c = document.createElementNS(ns, 'circle');
        c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', centre ? 28 : 22);
        g.appendChild(c);
        const t = document.createElementNS(ns, 'text');
        t.setAttribute('x', x); t.setAttribute('y', y + (centre ? 44 : 38));
        t.setAttribute('text-anchor', 'middle');
        t.textContent = name.length > 22 ? name.slice(0, 21) + '…' : name;
        const title = document.createElementNS(ns, 'title');
        title.textContent = name;
        g.appendChild(title);
        g.appendChild(t);
        svg.appendChild(g);
    };
    nodes.forEach(n => addNode(n.x, n.y, n.r.other_name || '?', n.r.other_uuid, false));
    addNode(cx, cy, centreName, null, true);

    el.innerHTML = '';
    el.appendChild(svg);
    if (rels.length > MAX) {
        const note = document.createElement('p');
        note.className = 'text-muted';
        note.textContent = `Graph shows the ${MAX} most recent of ${rels.length} relationships. The table lists all loaded rows.`;
        el.appendChild(note);
    }
}
