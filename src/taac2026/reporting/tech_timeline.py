"""Fetch recommendation-system paper lineage from Semantic Scholar and produce
an ECharts force-directed graph JSON file.

This module is the domain / reporting layer.  It contains:
- seed paper definitions with Semantic Scholar IDs
- API fetching helpers (stdlib ``urllib`` only — no extra dependencies)
- citation-graph expansion and filtering
- ECharts option generation

The companion CLI lives in
``taac2026.application.reporting.timeline_cli``.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Seed papers — each has a Semantic Scholar paper ID (corpus or SHA)
# and a manually assigned branch label for visualisation.
# ---------------------------------------------------------------------------

BRANCHES: list[str] = [
    "基础方法",
    "深度推荐",
    "序列建模",
    "特征交叉",
    "长序列",
    "统一建模",
    "生成式推荐",
]

BRANCH_INDEX: dict[str, int] = {b: i for i, b in enumerate(BRANCHES)}


@dataclass(slots=True)
class SeedPaper:
    """A seed paper that anchors the technology timeline."""

    query: str  # arXiv ID (e.g. "ArXiv:1706.06978") or title for search
    short_name: str
    branch: str
    highlight: bool = False


# Identifiers: prefer ArXiv:XXXX.XXXXX or DOI for stable lookup;
# these use the paper-lookup API (lenient rate limit) instead of the
# heavily rate-limited search API.  Title strings are kept as fallback.
SEED_PAPERS: list[SeedPaper] = [
    # --- 基础方法 ---
    SeedPaper("DOI:10.1145/371920.372071", "Item CF", "基础方法"),
    SeedPaper("DOI:10.1109/MC.2009.263", "MF", "基础方法"),
    SeedPaper("DOI:10.1109/ICDM.2010.127", "FM", "基础方法"),
    SeedPaper("DOI:10.1145/2505515.2505665", "DSSM", "基础方法"),
    # --- 深度推荐 ---
    SeedPaper("DOI:10.1145/2959100.2959190", "YouTube DNN", "深度推荐"),
    SeedPaper("ArXiv:1606.07792", "Wide & Deep", "深度推荐"),
    SeedPaper("ArXiv:1703.04247", "DeepFM", "深度推荐"),
    SeedPaper("ArXiv:1906.00091", "DLRM", "深度推荐"),
    # --- 序列建模 ---
    SeedPaper("DOI:10.48550/arxiv.1511.06939", "GRU4Rec", "序列建模"),
    SeedPaper("ArXiv:1706.06978", "DIN", "序列建模"),
    SeedPaper("ArXiv:1808.09781", "SASRec", "序列建模"),
    SeedPaper("ArXiv:1809.03672", "DIEN", "序列建模"),
    SeedPaper("ArXiv:1904.06690", "BERT4Rec", "序列建模"),
    # --- 特征交叉 ---
    SeedPaper("ArXiv:2008.13535", "DCNv2", "特征交叉"),
    SeedPaper("ArXiv:2203.11014", "DHEN", "特征交叉"),
    # --- 长序列 ---
    SeedPaper("ArXiv:2006.05639", "SIM", "长序列"),
    # --- 统一建模 ---
    SeedPaper("ArXiv:2402.17152", "HSTU", "统一建模"),
    SeedPaper("DOI:10.48550/arxiv.2411.09852", "InterFormer", "统一建模", highlight=True),
    SeedPaper("ArXiv:2601.12681", "HyFormer", "统一建模", highlight=True),
    SeedPaper("ArXiv:2510.26104", "OneTrans", "统一建模", highlight=True),
    SeedPaper("CorpusId:279155116", "GPSD", "统一建模"),
    SeedPaper("ArXiv:2508.02929", "Foundation-Expert", "统一建模"),
    SeedPaper("ArXiv:2510.11100", "HoMer", "统一建模"),
    SeedPaper("ArXiv:2510.15286", "MTmixAtt", "统一建模"),
    # --- 生成式推荐 ---
    SeedPaper("ArXiv:2307.00457", "GenRec", "生成式推荐"),
    SeedPaper("ArXiv:2403.19021", "IDGenRec", "生成式推荐"),
    SeedPaper("ArXiv:2403.10667", "UniMP", "生成式推荐"),
    SeedPaper("ArXiv:2502.10157", "SessionRec", "生成式推荐"),
    SeedPaper("ArXiv:2503.02453", "COBRA", "生成式推荐"),
    SeedPaper("ArXiv:2511.06254", "LLaDA-Rec", "生成式推荐"),
    SeedPaper("ArXiv:2512.14503", "RecGPT-V2", "生成式推荐"),
    SeedPaper("ArXiv:2512.11529", "xGR", "生成式推荐"),
    SeedPaper("ArXiv:2512.22386", "OxygenREC", "生成式推荐"),
    SeedPaper("ArXiv:2512.24787", "HiGR", "生成式推荐"),
]


def _cache_matches_query(data: dict[str, Any], query: str) -> bool:
    """Return whether cached metadata still matches the lookup query."""

    ext_ids = data.get("externalIds") or {}
    if query.startswith("ArXiv:"):
        return (ext_ids.get("ArXiv") or "").lower() == query.removeprefix(
            "ArXiv:"
        ).lower()
    if query.startswith("DOI:"):
        doi = query.removeprefix("DOI:").lower()
        if (ext_ids.get("DOI") or "").lower() == doi:
            return True
        arxiv_id = (ext_ids.get("ArXiv") or "").lower()
        return doi.startswith("10.48550/arxiv.") and arxiv_id == doi.removeprefix(
            "10.48550/arxiv."
        )

    def _norm(text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum())

    normalized_query = _norm(query)
    return bool(normalized_query) and normalized_query in _norm(data.get("title", ""))


def _has_complete_cache_entry(data: dict[str, Any], query: str) -> bool:
    """Return whether cached metadata is sufficient to skip re-resolution."""

    return (
        bool(data.get("year"))
        and bool(data.get("paperId"))
        and _cache_matches_query(data, query)
    )

_S2_API = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,year,citationCount,fieldsOfStudy,venue,authors,abstract,url,externalIds"
_S2_REF_FIELDS = "title,year,citationCount,isInfluential"
_S2_CITE_FIELDS = _S2_REF_FIELDS
_RATE_LIMIT_DELAY = 1.2  # seconds between requests (paper lookup: ~100 req/5min)
_SEARCH_RATE_DELAY = 3.5  # seconds between search requests (stricter limit)
_USER_AGENT = "TAAC2026-TimelineBot/1.0 (research; non-commercial)"
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_get(url: str, *, api_key: str | None = None) -> dict[str, Any]:
    """GET *url* and return parsed JSON.  Retries on 429 with exponential backoff."""
    headers = {"User-Agent": _USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 2 ** attempt * _RATE_LIMIT_DELAY
                time.sleep(wait)
                continue
            if exc.code == 404:
                return {}
            print(
                f"[tech_timeline] HTTP {exc.code} for {url}; falling back to cache-only mode.",
                file=sys.stderr,
            )
            return {}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            wait = 2 ** attempt * _RATE_LIMIT_DELAY
            if attempt < 4:
                print(
                    f"[tech_timeline] {type(exc).__name__} for {url}; retrying in {wait:.1f}s.",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(
                f"[tech_timeline] {type(exc).__name__} for {url}; falling back to cache-only mode.",
                file=sys.stderr,
            )
            return {}
    return {}


# ---------------------------------------------------------------------------
# Metadata cache — avoids re-fetching paper metadata from the API on every run.
# Stores full paper metadata, so cached papers need zero metadata API calls;
# reference/citation lookups for graph edges may still require API calls.
# ---------------------------------------------------------------------------

_DEFAULT_CACHE = _REPO_ROOT / ".cache/taac2026/.s2_cache.json"


def load_cache(path: Path = _DEFAULT_CACHE) -> dict[str, dict[str, Any]]:
    """Load ``{short_name: {paperId, year, citationCount, ...}}``."""
    if path.exists():
        text = path.read_text(encoding="utf-8-sig")  # tolerates BOM
        return json.loads(text)
    return {}


def save_cache(
    cache: dict[str, dict[str, Any]], path: Path = _DEFAULT_CACHE
) -> None:
    """Persist the metadata cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Paper lookup / search
# ---------------------------------------------------------------------------


def fetch_paper(
    paper_id: str, *, api_key: str | None = None
) -> dict[str, Any]:
    """Fetch core metadata for a single paper."""
    encoded_paper_id = urllib.parse.quote(paper_id, safe="")
    url = f"{_S2_API}/paper/{encoded_paper_id}?fields={_S2_FIELDS}"
    time.sleep(_RATE_LIMIT_DELAY)
    return _api_get(url, api_key=api_key)


def search_paper(
    title: str, *, api_key: str | None = None
) -> dict[str, Any]:
    """Search for a paper by title and return the best match."""
    encoded = urllib.parse.quote(title)
    url = f"{_S2_API}/paper/search?query={encoded}&limit=1&fields={_S2_FIELDS}"
    time.sleep(_SEARCH_RATE_DELAY)
    data = _api_get(url, api_key=api_key)
    papers = data.get("data", [])
    if papers:
        return papers[0]
    return {}


def resolve_paper(
    query: str, *, api_key: str | None = None
) -> dict[str, Any]:
    """Resolve a paper from an ArXiv ID, DOI, or title query.

    If *query* starts with ``ArXiv:`` or ``DOI:``, fetch by ID directly
    (uses the paper-lookup API with lenient rate limits).
    Otherwise, search by title (uses the search API with stricter limits).
    """
    if query.startswith(("ArXiv:", "DOI:", "CorpusId:")):
        return fetch_paper(query, api_key=api_key)
    return search_paper(query, api_key=api_key)


def fetch_references(
    paper_id: str,
    *,
    limit: int = 100,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return references (papers this paper cites)."""
    encoded_paper_id = urllib.parse.quote(paper_id, safe="")
    url = (
        f"{_S2_API}/paper/{encoded_paper_id}/references"
        f"?fields={_S2_REF_FIELDS}&limit={limit}"
    )
    time.sleep(_RATE_LIMIT_DELAY)
    data = _api_get(url, api_key=api_key)
    return data.get("data", [])


def fetch_citations(
    paper_id: str,
    *,
    limit: int = 100,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return citations (papers that cite this paper)."""
    encoded_paper_id = urllib.parse.quote(paper_id, safe="")
    url = (
        f"{_S2_API}/paper/{encoded_paper_id}/citations"
        f"?fields={_S2_CITE_FIELDS}&limit={limit}"
    )
    time.sleep(_RATE_LIMIT_DELAY)
    data = _api_get(url, api_key=api_key)
    return data.get("data", [])


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PaperNode:
    """A paper represented as a node in the tech-evolution graph."""

    s2_id: str
    name: str
    year: int
    branch: str
    citation_count: int = 0
    highlight: bool = False
    title: str = ""
    authors: str = ""  # comma-joined top authors
    venue: str = ""
    abstract: str = ""
    url: str = ""


@dataclass(slots=True)
class CitationEdge:
    """A directed edge: *source* influenced/was-cited-by *target*."""

    source: str  # short name
    target: str  # short name
    is_cross_branch: bool = False


@dataclass
class TimelineGraph:
    """Complete graph ready for ECharts serialisation."""

    nodes: list[PaperNode] = field(default_factory=list)
    edges: list[CitationEdge] = field(default_factory=list)
    _id_to_name: dict[str, str] = field(default_factory=dict)


def build_graph(
    *,
    seeds: list[SeedPaper] | None = None,
    expand_depth: int = 0,
    min_citations: int = 50,
    api_key: str | None = None,
    progress_callback: Any | None = None,
) -> TimelineGraph:
    """Build the technology timeline graph.

    Parameters
    ----------
    seeds:
        Override the default seed list.
    expand_depth:
        How many citation hops to expand (0 = seeds only, 1 = one hop).
    min_citations:
        Minimum citation count for expanded papers to be included.
    api_key:
        Optional Semantic Scholar API key for higher rate limits.
    progress_callback:
        ``callback(msg: str)`` called for progress reporting.
    """
    if seeds is None:
        seeds = SEED_PAPERS

    graph = TimelineGraph()
    pending: list[tuple[str, str, str, bool]] = [
        (s.query, s.short_name, s.branch, s.highlight) for s in seeds
    ]

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # Phase 1 — resolve seed papers via API (with full metadata caching)
    cache = load_cache()
    cached_count = len(cache)
    _log(
        f"Fetching {len(pending)} seed papers (cache has {cached_count} entries) …"
    )
    cache_dirty = False
    for query, short_name, branch, highlight in pending:
        # Prefer cached metadata when it is complete. If the cache matches the
        # query but lacks a paperId, keep docs generation deterministic by
        # using the cached metadata directly unless an API key is available to
        # refresh it.
        cached = cache.get(short_name)
        if cached and _has_complete_cache_entry(cached, query):
            data = cached
        elif cached and cached.get("year") and _cache_matches_query(cached, query) and not api_key:
            data = cached
            _log(f"  ⚠ {short_name}: using incomplete cached metadata (cache-only mode)")
        else:
            data = resolve_paper(query, api_key=api_key)
            if data and "year" in data:
                cache[short_name] = data
                cache_dirty = True
                # Save incrementally so progress survives crashes
                save_cache(cache)
            elif cached and cached.get("year") and _cache_matches_query(cached, query):
                data = cached
                _log(f"  ⚠ {short_name}: using incomplete cached metadata")
        s2_id = data.get("paperId", "")
        if not data or "year" not in data:
            _log(f"  ⚠ {short_name}: not found ({query})")
            continue
        # Extract author names (top 5)
        raw_authors = data.get("authors") or []
        author_names = [a.get("name", "") for a in raw_authors[:5]]
        if len(raw_authors) > 5:
            author_names.append("et al.")
        authors_str = ", ".join(author_names)

        # Build S2 or ArXiv URL
        ext_ids = data.get("externalIds") or {}
        paper_url = data.get("url", "")
        if not paper_url and ext_ids.get("ArXiv"):
            paper_url = f"https://arxiv.org/abs/{ext_ids['ArXiv']}"
        if not paper_url and ext_ids.get("DOI"):
            paper_url = f"https://doi.org/{ext_ids['DOI']}"

        # Truncate abstract
        abstract_raw = data.get("abstract") or ""
        abstract_short = (
            abstract_raw[:200] + "…" if len(abstract_raw) > 200 else abstract_raw
        )

        node = PaperNode(
            s2_id=s2_id,
            name=short_name,
            year=data.get("year", 0),
            branch=branch,
            citation_count=data.get("citationCount", 0),
            highlight=highlight,
            title=data.get("title", ""),
            authors=authors_str,
            venue=data.get("venue", ""),
            abstract=abstract_short,
            url=paper_url,
        )
        graph.nodes.append(node)
        if s2_id:
            graph._id_to_name[s2_id] = short_name
        _log(f"  ✓ {short_name} ({node.year}, {node.citation_count} cites)")
    if cache_dirty:
        save_cache(cache)
        _log(f"  Cache: {cached_count} → {len(cache)} entries")

    # Phase 2 — discover edges among seed papers via references
    _log("Discovering citation edges among seed papers …")
    known_ids = {paper_id for paper_id in graph._id_to_name if paper_id}
    for node in list(graph.nodes):
        if not node.s2_id:
            continue
        refs = fetch_references(node.s2_id, api_key=api_key)
        if refs is None:
            refs = []
        for ref in refs:
            cited = ref.get("citedPaper") or {}
            cited_id = cited.get("paperId", "")
            if cited_id in known_ids:
                src_name = graph._id_to_name[cited_id]
                tgt_name = node.name
                src_branch = _branch_of(graph, src_name)
                tgt_branch = _branch_of(graph, tgt_name)
                edge = CitationEdge(
                    source=src_name,
                    target=tgt_name,
                    is_cross_branch=(src_branch != tgt_branch),
                )
                graph.edges.append(edge)
        _log(f"  {node.name}: checked {len(refs)} references")

    # Phase 3 — optional expansion
    if expand_depth >= 1:
        _log("Expanding one hop from seed papers …")
        for node in list(graph.nodes):
            if not node.s2_id:
                continue
            cites = fetch_citations(
                node.s2_id, limit=50, api_key=api_key
            )
            if cites is None:
                cites = []
            for cite_entry in cites:
                citing = cite_entry.get("citingPaper", {})
                if not citing:
                    continue
                citing_id = citing.get("paperId", "")
                if not citing_id or citing_id in known_ids:
                    continue
                cc = citing.get("citationCount", 0)
                year = citing.get("year")
                if cc < min_citations or not year:
                    continue
                is_influential = cite_entry.get("isInfluential", False)
                if not is_influential:
                    continue
                title = citing.get("title", "")
                short = _dedupe_name(
                    {existing.name for existing in graph.nodes},
                    _shorten(title),
                )
                branch = node.branch  # inherit parent branch
                new_node = PaperNode(
                    s2_id=citing_id,
                    name=short,
                    year=year,
                    branch=branch,
                    citation_count=cc,
                )
                graph.nodes.append(new_node)
                graph._id_to_name[citing_id] = short
                known_ids.add(citing_id)
                graph.edges.append(
                    CitationEdge(
                        source=node.name,
                        target=short,
                        is_cross_branch=False,
                    )
                )
                _log(f"  + {short} ({year}, {cc} cites)")

    # Deduplicate edges
    seen_edges: set[tuple[str, str]] = set()
    unique_edges: list[CitationEdge] = []
    for e in graph.edges:
        key = (e.source, e.target)
        if key not in seen_edges and e.source != e.target:
            seen_edges.add(key)
            unique_edges.append(e)
    graph.edges = unique_edges

    _log(f"Done: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


def _branch_of(graph: TimelineGraph, name: str) -> str:
    for n in graph.nodes:
        if n.name == name:
            return n.branch
    return ""


def _shorten(title: str) -> str:
    """Extract a short identifier from a paper title."""
    # Try to find a CamelCase acronym or model name
    parts = title.split(":")
    candidate = parts[0].strip()
    if len(candidate) <= 20:
        return candidate
    # Fallback: first 3 significant words
    words = [w for w in candidate.split() if len(w) > 2]
    return " ".join(words[:3])


def _dedupe_name(existing_names: set[str], preferred: str) -> str:
    """Return a display name that is unique within *existing_names*."""
    if preferred not in existing_names:
        return preferred
    suffix = 2
    while True:
        candidate = f"{preferred} #{suffix}"
        if candidate not in existing_names:
            return candidate
        suffix += 1


# ---------------------------------------------------------------------------
# ECharts serialisation
# ---------------------------------------------------------------------------

# Curated palette — one colour per branch, distinguishable on both
# light and dark backgrounds.
_BRANCH_COLORS: list[str] = [
    "#5470c6",  # 基础方法  — blue
    "#91cc75",  # 深度推荐  — green
    "#fac858",  # 序列建模  — amber
    "#ee6666",  # 特征交叉  — red
    "#73c0de",  # 长序列    — cyan
    "#3ba272",  # 统一建模  — teal
    "#fc8452",  # 生成式推荐 — orange
]


def _symbol_size(citation_count: int) -> int:
    """Map citation count to a node radius."""
    if citation_count >= 5000:
        return 36
    if citation_count >= 2000:
        return 30
    if citation_count >= 500:
        return 24
    if citation_count >= 100:
        return 18
    return 14


def to_echarts(graph: TimelineGraph) -> dict[str, Any]:
    """Convert *graph* to an ECharts force-directed graph option dict."""
    # ── build node list ──
    # Use year as initial x hint so the force layout starts roughly
    # chronological (left→right).  y is seeded by branch index so
    # same-branch papers cluster vertically.
    years = [n.year for n in graph.nodes if n.year > 0]
    min_year = min(years) if years else 2000
    max_year = max(years) if years else 2026
    yr_span = max(max_year - min_year, 1)

    nodes_data: list[dict[str, Any]] = []
    node_key_by_name: dict[str, str] = {}
    for n in graph.nodes:
        if n.year <= 0:
            continue
        y_idx = BRANCH_INDEX.get(n.branch, 0)
        node_key = n.s2_id or n.name
        node_key_by_name[n.name] = node_key
        # Spread initial positions across a 800×500 canvas
        x_init = (n.year - min_year) / yr_span * 700 + 50
        y_init = y_idx * 70 + 40

        node_dict: dict[str, Any] = {
            "id": node_key,
            "name": n.name,
            "x": x_init,
            "y": y_init,
            "symbolSize": _symbol_size(n.citation_count),
            "category": y_idx,
            "label": {
                "show": True,
                "fontSize": 11,
                "fontWeight": "bold",
            },
            "title": n.title,
            "authors": n.authors,
            "venue": n.venue,
            "paperYear": n.year,
            "citations": n.citation_count,
            "abstract": n.abstract,
            "paperUrl": n.url,
        }
        if n.highlight:
            node_dict["itemStyle"] = {
                "borderColor": "#FFD700",
                "borderWidth": 3,
            }
            node_dict["label"]["fontSize"] = 12
        nodes_data.append(node_dict)

    # ── build links (all edges, no filtering) ──
    links_data: list[dict[str, Any]] = []
    for e in graph.edges:
        source_id = node_key_by_name.get(e.source)
        target_id = node_key_by_name.get(e.target)
        if not source_id or not target_id:
            continue
        style: dict[str, Any] = {"width": 1.5, "opacity": 0.5}
        if e.is_cross_branch:
            style.update(type="dashed", opacity=0.3)
        links_data.append({
            "source": source_id,
            "target": target_id,
            "sourceName": e.source,
            "targetName": e.target,
            "lineStyle": style,
        })

    # ── categories ──
    categories = [
        {"name": b, "itemStyle": {"color": _BRANCH_COLORS[i]}}
        for i, b in enumerate(BRANCHES)
    ]

    option: dict[str, Any] = {
        "_height": "640px",
        "tooltip": {"trigger": "item"},
        "legend": {
            "data": BRANCHES,
            "top": 0,
            "itemWidth": 14,
            "itemHeight": 14,
            "textStyle": {"fontSize": 12},
        },
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "force": {
                    "repulsion": 350,
                    "edgeLength": [80, 200],
                    "gravity": 0.12,
                    "layoutAnimation": True,
                },
                "roam": True,
                "draggable": True,
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [4, 8],
                "label": {
                    "show": True,
                    "position": "right",
                    "fontSize": 11,
                    "fontWeight": "bold",
                },
                "lineStyle": {
                    "color": "#999",
                    "width": 1.5,
                    "curveness": 0.15,
                    "opacity": 0.4,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "label": {"fontSize": 14},
                    "lineStyle": {"width": 3, "opacity": 1},
                },
                "animationDuration": 1500,
                "categories": categories,
                "data": nodes_data,
                "links": links_data,
            }
        ],
    }
    return option


def write_echarts_json(option: dict[str, Any], path: Path) -> None:
    """Write *option* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(option, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
