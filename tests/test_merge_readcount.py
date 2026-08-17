"""Golden test for merge-sample-bam-readcount.py.

The parser was rewritten to stream through Arrow instead of accumulating a dict
per readcount row. This pins the new output against the original list-of-dicts
implementation, kept below as the reference, so the rewrite can't quietly change
which bases get emitted or dropped.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "merge-sample-bam-readcount.py"


@pytest.fixture
def merge():
    spec = importlib.util.spec_from_file_location("merge_readcount", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# values[5]/[6]/[10] are the strand and q2 read counts -- integers, unlike the
# quality metrics around them.
INT_FIELDS = {5, 6, 10}


def field(base, count):
    """A bam-readcount base field: base:count: + 12 more metrics."""
    metrics = [str(i) if i in INT_FIELDS else f"{i}.5" for i in range(2, 14)]
    return ":".join([base, str(count), *metrics])


def line(pos, extras=(), depth=50):
    cols = [
        "chr1",
        str(pos),
        "A",
        str(depth),
        field("=", 0),
        field("A", 30),  # kept
        field("C", 0),  # dropped: zero count
        field("G", 7),  # kept
        field("T", 0),  # dropped: zero count
        *extras,
    ]
    return "\t".join(cols) + "\n"


# --- original implementation, verbatim apart from the shared record() helper ---
def old_process_file(file_path, sample):
    readcount_dict = []
    with open(file_path) as file:
        for text in file:
            row = text.rstrip().split("\t")
            chrom, pos, ref, depth, base_equals, base_A, base_C, base_G, base_T = row[0:9]
            del row[0:9]
            bases = {"A": base_A, "C": base_C, "G": base_G, "T": base_T}

            def record(base, values):
                return {
                    "sample": sample,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "depth": depth,
                    "base": base,
                    "count": int(values[1]),
                    "avg_mapping_quality": float(values[2]),
                    "avg_basequality": float(values[3]),
                    "avg_se_mapping_quality": float(values[4]),
                    "num_plus_strand": int(values[5]),
                    "num_minus_strand": int(values[6]),
                    "avg_pos_as_fraction": float(values[7]),
                    "avg_num_mismatches_as_fraction": float(values[8]),
                    "avg_sum_mismatch_qualities": float(values[9]),
                    "num_q2_containing_reads": int(values[10]),
                    "avg_distance_to_q2_start_in_q2_reads": float(values[11]),
                    "avg_clipped_length": float(values[12]),
                    "avg_distance_to_effective_3p_end": float(values[13]),
                }

            for base, value in bases.items():
                values = value.split(":")
                if int(values[1]) == 0:
                    continue
                readcount_dict.append(record(base, values))

            for i in row:
                if i[0] == "N":
                    continue
                values = i.split(":")
                readcount_dict.append(record(values[0], values))
    return readcount_dict


def old_output(snv, indel, sample, dtypes):
    df = pd.DataFrame(columns=list(dtypes.keys()))
    rows = old_process_file(snv, sample) + old_process_file(indel, sample)
    df = pd.concat([df, pd.DataFrame.from_dict(rows)], ignore_index=True).drop_duplicates()
    return df.astype(dtypes)


def test_matches_original_implementation(merge, tmp_path, monkeypatch):
    # small enough to force several Arrow chunks out of a handful of rows
    monkeypatch.setattr(merge, "CHUNK_ROWS", 2)

    snv = tmp_path / "snv.tsv"
    indel = tmp_path / "indel.tsv"
    out = tmp_path / "out.parquet"

    snv.write_text(
        line(100)
        + line(200, extras=[field("N", 3)])  # N is dropped
        + line(200, extras=[field("N", 3)])  # repeat within one file: collapsed
        + line(300, extras=[field("+AG", 0)])  # zero-count indel is NOT dropped
    )
    indel.write_text(
        line(100)  # repeats the snv line exactly: collapsed
        + line(300, depth=77)  # same position as snv, different depth: BOTH kept
        + line(400, extras=[field("-CT", 5)])
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["merge", "--sample", "s1", "--snv", str(snv), "--indel", str(indel), "--out", str(out)],
    )
    merge.main()

    new = pd.read_parquet(out).reset_index(drop=True)
    expected = old_output(str(snv), str(indel), "s1", merge.DTYPES).reset_index(drop=True)

    assert list(new["base"]) == [
        "A", "G",           # pos 100 (the indel file's copy collapsed into this)
        "A", "G",           # pos 200 (the repeat within snv collapsed)
        "A", "G", "+AG",    # pos 300, depth 50
        "A", "G",           # pos 300 again at depth 77 -- NOT a duplicate
        "A", "G", "-CT",    # pos 400
    ]
    assert list(new["depth"]) == [50] * 7 + [77] * 2 + [50] * 3
    # check_categorical is on: the Arrow dictionaries must end up with the same
    # category order pandas' own astype('category') would have produced.
    pd.testing.assert_frame_equal(new, expected)


def test_rejects_truncated_line(merge, tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("chr1\t100\tA\t50\n")
    with pytest.raises(ValueError, match="expected at least 9"):
        list(merge.iter_rows(str(bad), "s1"))
