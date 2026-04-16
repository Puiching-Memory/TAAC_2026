from __future__ import annotations

import json

from taac2026.reporting import tech_timeline

CitationEdge = tech_timeline.CitationEdge
PaperNode = tech_timeline.PaperNode
SeedPaper = tech_timeline.SeedPaper
TimelineGraph = tech_timeline.TimelineGraph
build_graph = tech_timeline.build_graph
to_echarts = tech_timeline.to_echarts


class TestTechTimelineECharts:
    def test_to_echarts_returns_serializable_graph_option(self) -> None:
        graph = TimelineGraph(
            nodes=[
                PaperNode(
                    s2_id="seed-1",
                    name="Seed",
                    year=2024,
                    branch="生成式推荐",
                    citation_count=128,
                    title="Seed paper",
                    authors="A. Author",
                    venue="arXiv",
                    abstract="Demo abstract",
                    url="https://example.com/paper",
                ),
                PaperNode(
                    s2_id="seed-2",
                    name="FollowUp",
                    year=2025,
                    branch="生成式推荐",
                    citation_count=256,
                ),
            ],
            edges=[CitationEdge(source="Seed", target="FollowUp")],
        )

        option = to_echarts(graph)

        assert option["_height"] == "640px"
        assert option["series"][0]["type"] == "graph"
        assert option["series"][0]["data"]
        assert option["series"][0]["links"] == [
            {
                "source": "Seed",
                "target": "FollowUp",
                "lineStyle": {"width": 1.5, "opacity": 0.5},
            }
        ]
        assert '"type": "graph"' in json.dumps(option, ensure_ascii=False)


class TestTechTimelineApiPaths:
    def test_fetch_paper_url_encodes_doi_path_segment(self, monkeypatch) -> None:
        captured: dict[str, str] = {}

        def _fake_api_get(url: str, *, api_key: str | None = None) -> dict[str, object]:
            captured["url"] = url
            captured["api_key"] = api_key or ""
            return {}

        monkeypatch.setattr(tech_timeline, "_api_get", _fake_api_get)
        monkeypatch.setattr(tech_timeline.time, "sleep", lambda _: None)

        tech_timeline.fetch_paper(
            "DOI:10.1145/371920.372071",
            api_key="demo-key",
        )

        assert (
            captured["url"]
            == "https://api.semanticscholar.org/graph/v1/paper/"
            "DOI%3A10.1145%2F371920.372071"
            "?fields=title,year,citationCount,fieldsOfStudy,venue,authors,abstract,url,externalIds"
        )
        assert captured["api_key"] == "demo-key"


class TestTechTimelineCacheFallback:
    def test_build_graph_re_resolves_incomplete_cache_entry(self, monkeypatch) -> None:
        monkeypatch.setattr(
            tech_timeline,
            "load_cache",
            lambda path=None: {
                "IDGenRec": {
                    "paperId": "",
                    "externalIds": {"ArXiv": "2403.19021"},
                    "title": "IDGenRec: stale cache",
                    "year": 2024,
                    "citationCount": 1,
                }
            },
        )
        monkeypatch.setattr(
            tech_timeline,
            "resolve_paper",
            lambda query, api_key=None: {
                "paperId": "resolved-paper",
                "externalIds": {"ArXiv": "2403.19021"},
                "title": "IDGenRec: LLM-RecSys Alignment with Textual ID Learning",
                "year": 2024,
                "citationCount": 42,
                "authors": [],
                "venue": "SIGIR",
                "abstract": "",
                "url": "https://example.com/idgenrec",
            },
        )
        monkeypatch.setattr(tech_timeline, "save_cache", lambda *args, **kwargs: None)
        monkeypatch.setattr(tech_timeline, "fetch_references", lambda *args, **kwargs: [])
        monkeypatch.setattr(tech_timeline, "fetch_citations", lambda *args, **kwargs: [])

        graph = build_graph(
            seeds=[
                SeedPaper(
                    query="ArXiv:2403.19021",
                    short_name="IDGenRec",
                    branch="生成式推荐",
                )
            ]
        )

        assert [node.s2_id for node in graph.nodes] == ["resolved-paper"]
        assert graph._id_to_name == {"resolved-paper": "IDGenRec"}
        assert "" not in graph._id_to_name
