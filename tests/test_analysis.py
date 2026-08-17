import numpy as np
import pandas as pd
import pytest

from analysis import add_metrics, classify_readcount_type, parse_variants


def test_parse_variants_snp():
    df = parse_variants(["chr1:100:A>T"])
    row = df.iloc[0]
    assert row["chrom"] == "chr1"
    assert row["pos"] == 100
    assert row["readcount_ref"] == "A"
    assert row["readcount_alt"] == "T"
    assert row["variant_type"] == "snp"


def test_parse_variants_deletion_shifts_past_anchor():
    row = parse_variants(["chr2:200:TAG>T"]).iloc[0]
    assert row["pos"] == 201
    assert row["readcount_ref"] == "A"
    assert row["readcount_alt"] == "-AG"
    assert row["variant_type"] == "del"


def test_parse_variants_insertion_keeps_pos():
    row = parse_variants(["chr3:300:C>CAG"]).iloc[0]
    assert row["pos"] == 300
    assert row["readcount_ref"] == "C"
    assert row["readcount_alt"] == "+AG"
    assert row["variant_type"] == "ins"


def test_parse_variants_rejects_mnp():
    with pytest.raises(ValueError, match="mnp"):
        parse_variants(["chr1:100:AT>GC"])


def test_classify_readcount_type():
    matches = pd.DataFrame(
        {
            "base": ["T", "A", "G"],
            "readcount_alt": ["T", "T", "T"],
            "readcount_ref": ["A", "A", "A"],
        }
    )
    assert list(classify_readcount_type(matches)) == ["alt", "ref", "offsite"]


def _long_row(sample, readcount_type, base_fraction, avg_pos_as_fraction):
    return {
        "sample": sample,
        "chrom": "chr1",
        "pos": 100,
        "ref": "A",
        "variant": "chr1:100:A>T",
        "variant_type": "snp",
        "depth": 1000,
        "variant_called": True,
        "readcount_type": readcount_type,
        "base": "T" if readcount_type == "alt" else "A",
        "count": int(base_fraction * 1000),
        "base_fraction": base_fraction,
        "avg_pos_as_fraction": avg_pos_as_fraction,
        "avg_mapping_quality": 60.0,
        "avg_basequality": 30.0,
        "avg_se_mapping_quality": 60.0,
        "num_plus_strand": 10,
        "num_minus_strand": 10,
        "plus_strand_bias": 0.5,
        "avg_num_mismatches_as_fraction": 0.01,
        "avg_sum_mismatch_qualities": 30.0,
        "num_q2_containing_reads": 0,
        "avg_distance_to_q2_start_in_q2_reads": 0.0,
        "avg_clipped_length": 100.0,
        "avg_distance_to_effective_3p_end": 0.5,
        "callers": "callerA",
        "callers_filter": "PASS",
        "callers_count": 1,
    }


def test_add_metrics_zscore_and_delta():
    rows = []
    for sample, alt_frac in [("s1", 0.1), ("s2", 0.2), ("s3", 0.3)]:
        rows.append(_long_row(sample, "alt", alt_frac, 0.3))
        rows.append(_long_row(sample, "ref", 1 - alt_frac, 0.5))
        rows.append(_long_row(sample, "offsite", 0.01, 0.5))
    wide = add_metrics(pd.DataFrame(rows))

    assert len(wide) == 3  # one row per sample; offsite rows dropped
    wide = wide.sort_values("sample").reset_index(drop=True)
    # base_fraction 0.1/0.2/0.3 across samples: z = -1, 0, 1
    np.testing.assert_allclose(wide["zscore_alt"], [-1.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(wide["avg_pos_as_fraction_delta"], [0.2] * 3)
    assert set(["base_fraction_alt", "count_ref", "count_alt"]) <= set(wide.columns)
