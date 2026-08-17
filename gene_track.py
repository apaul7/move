import re

import requests

UCSC_API_BASE = "https://api.genome.ucsc.edu"


def search_gene_region(gene_symbol, genome="hg38", timeout=10):
    resp = requests.get(
        f"{UCSC_API_BASE}/search",
        params={"search": gene_symbol, "genome": genome},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    for group in data.get("positionMatches", []):
        if group.get("name") != "hgnc":
            continue
        for match in group.get("matches", []):
            if match.get("posName", "").lower() == gene_symbol.lower():
                return _parse_position(match.get("position"))
    return None


def _parse_position(position):
    if not position:
        return None
    m = re.match(r"^(\S+):(\d+)-(\d+)$", position)
    if not m:
        return None
    chrom, start, end = m.groups()
    return chrom, int(start), int(end)


def fetch_transcripts(
    gene_symbol,
    chrom,
    start,
    end,
    genome="hg38",
    track="wgEncodeGencodeBasicV46",
    timeout=10,
):
    resp = requests.get(
        f"{UCSC_API_BASE}/getData/track",
        params={
            "genome": genome,
            "track": track,
            "chrom": chrom,
            "start": start,
            "end": end,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    transcripts = data.get(track, [])
    return [
        t for t in transcripts if t.get("name2", "").lower() == gene_symbol.lower()
    ]


def strip_transcript_version(transcript_id):
    return transcript_id.split(".")[0]


def select_transcript(transcripts, preferred_transcript_id=None):
    if not transcripts:
        return None

    if preferred_transcript_id:
        preferred_base = strip_transcript_version(preferred_transcript_id)
        for t in transcripts:
            if strip_transcript_version(t.get("name", "")) == preferred_base:
                return t

    return max(transcripts, key=lambda t: t.get("exonCount", 0))


def parse_exons(transcript):
    starts = [int(x) for x in transcript["exonStarts"].strip(",").split(",")]
    ends = [int(x) for x in transcript["exonEnds"].strip(",").split(",")]
    cds_start = transcript["cdsStart"]
    cds_end = transcript["cdsEnd"]

    exons = []
    for i, (start, end) in enumerate(zip(starts, ends), start=1):
        for piece_start, piece_end, region in _split_by_cds(
            start, end, cds_start, cds_end
        ):
            exons.append(
                {
                    "exon_number": i,
                    "start": piece_start,
                    "end": piece_end,
                    "region": region,
                }
            )
    return exons


def _split_by_cds(start, end, cds_start, cds_end):
    """Split an exon [start, end) into utr5/cds/utr3 pieces by CDS boundary.

    Doesn't care which side is 5' vs 3' (that depends on strand) -- pieces
    before cds_start are "utr5", inside are "cds", after cds_end are "utr3".
    Callers on the minus strand just get the label meaning flipped, which is
    fine since region is only used for exon height/color, not orientation.
    """
    pieces = []
    if end <= cds_start or start >= cds_end:
        region = "utr5" if end <= cds_start else "utr3"
        pieces.append((start, end, region))
        return pieces

    if start < cds_start:
        pieces.append((start, cds_start, "utr5"))
    pieces.append((max(start, cds_start), min(end, cds_end), "cds"))
    if end > cds_end:
        pieces.append((cds_end, end, "utr3"))
    return pieces


def build_display_layout(transcript, exon_display_width=40, intron_display_width=12):
    starts = [int(x) for x in transcript["exonStarts"].strip(",").split(",")]
    ends = [int(x) for x in transcript["exonEnds"].strip(",").split(",")]
    exons = parse_exons(transcript)

    # segments alternate: exon, intron, exon, intron, ..., exon -- built from
    # the un-split raw exon boundaries so introns are the gaps between them.
    exon_boxes = []
    breakpoints = []  # (genomic_start, genomic_end, disp_start, disp_end)
    cursor = 0.0

    for i, (raw_start, raw_end) in enumerate(zip(starts, ends)):
        disp_start = cursor
        disp_end = cursor + exon_display_width
        breakpoints.append((raw_start, raw_end, disp_start, disp_end))
        cursor = disp_end

        if i < len(starts) - 1:
            intron_start = raw_end
            intron_end = starts[i + 1]
            intron_disp_start = cursor
            intron_disp_end = cursor + intron_display_width
            breakpoints.append(
                (intron_start, intron_end, intron_disp_start, intron_disp_end)
            )
            cursor = intron_disp_end

    for piece in exons:
        for raw_start, raw_end, disp_start, disp_end in breakpoints:
            if piece["start"] >= raw_start and piece["end"] <= raw_end:
                frac_start = _fraction(piece["start"], raw_start, raw_end)
                frac_end = _fraction(piece["end"], raw_start, raw_end)
                exon_boxes.append(
                    {
                        "exon_number": piece["exon_number"],
                        "region": piece["region"],
                        "disp_start": disp_start + frac_start * (disp_end - disp_start),
                        "disp_end": disp_start + frac_end * (disp_end - disp_start),
                    }
                )
                break

    def genomic_to_display(pos):
        for raw_start, raw_end, disp_start, disp_end in breakpoints:
            if raw_start <= pos <= raw_end:
                frac = _fraction(pos, raw_start, raw_end)
                return disp_start + frac * (disp_end - disp_start)
        # outside the transcript's own span (e.g. a variant just upstream/
        # downstream of the first/last exon) -- clamp to the nearest edge.
        first_start = breakpoints[0][2]
        last_end = breakpoints[-1][3]
        return first_start if pos < starts[0] else last_end

    return exon_boxes, genomic_to_display


def _fraction(pos, start, end):
    if end == start:
        return 0.0
    return (pos - start) / (end - start)


PROTEIN_POSITION_RE = re.compile(r"\d+")


def parse_protein_position(raw):
    if raw is None:
        return None
    text = str(raw)
    m = PROTEIN_POSITION_RE.search(text)
    if not m:
        return None
    return int(m.group())
