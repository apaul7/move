import math

import pytest
import requests

import gene_track


# Loosely modeled on a real UCSC wgEncodeGencodeBasicV46 response for TP53
# (minus strand, 7 exons, single-exon straddling the CDS/UTR boundary).
TRANSCRIPT = {
    "name": "ENST00000413465.6",
    "chrom": "chr17",
    "strand": "-",
    "txStart": 100,
    "txEnd": 1000,
    "cdsStart": 150,
    "cdsEnd": 900,
    "exonCount": 3,
    "exonStarts": "100,400,850,",
    "exonEnds": "200,500,1000,",
    "name2": "TP53",
}

OTHER_TRANSCRIPT = {
    "name": "ENST00000999999.2",
    "chrom": "chr17",
    "strand": "-",
    "txStart": 100,
    "txEnd": 1000,
    "cdsStart": 150,
    "cdsEnd": 900,
    "exonCount": 2,
    "exonStarts": "100,850,",
    "exonEnds": "200,1000,",
    "name2": "TP53",
}


def test_select_transcript_matches_preferred_ignoring_version():
    chosen = gene_track.select_transcript(
        [OTHER_TRANSCRIPT, TRANSCRIPT], preferred_transcript_id="ENST00000413465.9"
    )
    assert chosen["name"] == "ENST00000413465.6"


def test_select_transcript_falls_back_to_most_exons():
    chosen = gene_track.select_transcript(
        [OTHER_TRANSCRIPT, TRANSCRIPT], preferred_transcript_id=None
    )
    assert chosen["name"] == "ENST00000413465.6"


def test_select_transcript_falls_back_when_preferred_not_found():
    chosen = gene_track.select_transcript(
        [OTHER_TRANSCRIPT], preferred_transcript_id="ENST00000413465.6"
    )
    assert chosen["name"] == "ENST00000999999.2"


def test_select_transcript_empty_list():
    assert gene_track.select_transcript([], preferred_transcript_id="x") is None


def test_parse_exons_classifies_utr_and_cds():
    exons = gene_track.parse_exons(TRANSCRIPT)

    # exon 1: [100, 200) vs cds [150, 900) -> split into utr5 + cds
    exon_1_pieces = [e for e in exons if e["exon_number"] == 1]
    assert {(p["start"], p["end"], p["region"]) for p in exon_1_pieces} == {
        (100, 150, "utr5"),
        (150, 200, "cds"),
    }

    # exon 2: [400, 500) fully inside cds [150, 900) -> single cds piece
    exon_2_pieces = [e for e in exons if e["exon_number"] == 2]
    assert exon_2_pieces == [
        {"exon_number": 2, "start": 400, "end": 500, "region": "cds"}
    ]

    # exon 3: [850, 1000) vs cds end 900 -> split into cds + utr3
    exon_3_pieces = [e for e in exons if e["exon_number"] == 3]
    assert {(p["start"], p["end"], p["region"]) for p in exon_3_pieces} == {
        (850, 900, "cds"),
        (900, 1000, "utr3"),
    }


def test_build_display_layout_orders_exons_and_introns():
    exon_boxes, genomic_to_display = gene_track.build_display_layout(
        TRANSCRIPT, exon_display_width=40, intron_display_width=12
    )

    # 3 exons, one split into two by the CDS boundary each on exon 1 and 3 ->
    # 5 display pieces total, monotonically increasing.
    assert len(exon_boxes) == 5
    starts = [b["disp_start"] for b in exon_boxes]
    assert starts == sorted(starts)

    # a position inside exon 2 (fully cds, [400, 500)) maps within exon 2's
    # display span, which starts after exon 1 (40) + one intron gap (12).
    mid_exon_2 = genomic_to_display(450)
    exon_2_disp_start = 40 + 12
    assert exon_2_disp_start <= mid_exon_2 <= exon_2_disp_start + 40

    # a position inside the intron between exon 1 and exon 2 ([200, 400))
    # maps into the compressed intron gap, not the full genomic span.
    mid_intron = genomic_to_display(300)
    assert 40 <= mid_intron <= 40 + 12


def test_build_display_layout_clamps_outside_transcript():
    exon_boxes, genomic_to_display = gene_track.build_display_layout(TRANSCRIPT)
    assert genomic_to_display(0) == exon_boxes[0]["disp_start"]
    assert genomic_to_display(5000) == exon_boxes[-1]["disp_end"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("123", 123),
        ("123-125", 123),
        ("?-125", 125),
        ("125-?", 125),
        (None, None),
        (float("nan"), None),
        ("", None),
    ],
)
def test_parse_protein_position(raw, expected):
    if raw is not None and isinstance(raw, float) and math.isnan(raw):
        assert gene_track.parse_protein_position(raw) is None
        return
    assert gene_track.parse_protein_position(raw) == expected


SEARCH_RESPONSE = {
    "positionMatches": [
        {
            "name": "hgnc",
            "matches": [
                {"position": "chr17:7661779-7687546", "posName": "TP53"},
                {"position": "chr2:24077433-24085861", "posName": "TP53I3"},
            ],
        }
    ]
}

TRACK_RESPONSE = {
    "track": "wgEncodeGencodeBasicV46",
    "wgEncodeGencodeBasicV46": [TRANSCRIPT, OTHER_TRANSCRIPT],
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_search_gene_region_parses_matching_hgnc_entry(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse(SEARCH_RESPONSE)
    )
    assert gene_track.search_gene_region("TP53") == ("chr17", 7661779, 7687546)


def test_search_gene_region_no_match(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _FakeResponse({"positionMatches": []})
    )
    assert gene_track.search_gene_region("NOPE") is None


def test_search_gene_region_propagates_network_errors(monkeypatch):
    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr(requests, "get", raise_timeout)
    with pytest.raises(requests.exceptions.Timeout):
        gene_track.search_gene_region("TP53")


def test_fetch_transcripts_filters_to_gene_symbol(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(TRACK_RESPONSE))
    result = gene_track.fetch_transcripts("TP53", "chr17", 7661779, 7687546)
    assert {t["name"] for t in result} == {TRANSCRIPT["name"], OTHER_TRANSCRIPT["name"]}


def test_fetch_transcripts_excludes_other_genes(monkeypatch):
    other_gene = {**TRANSCRIPT, "name": "ENST0000000000.1", "name2": "TP53I3"}
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse(
            {"track": "wgEncodeGencodeBasicV46", "wgEncodeGencodeBasicV46": [other_gene]}
        ),
    )
    result = gene_track.fetch_transcripts("TP53", "chr17", 7661779, 7687546)
    assert result == []
