import io
from datetime import datetime

import pandas as pd
import pytest

from filtering import build_export_csv, filter_results, samples_matching


def make_row(**overrides):
    row = {
        "variant": "chr1:100:A>T",
        "variant_called": True,
        "gnomADw4.1_AF": 0.01,
        "base_fraction_alt": 0.10,
        "depth": 1000,
        "count_alt": 100,
        "avg_pos_as_fraction_delta": 0.05,
        "plus_strand_bias_alt": 0.5,
        "zscore_alt": 10.0,
        "soft_masked": False,
        "callers_filter_alt": "PASS",
        "SYMBOL": "BRCA1",
    }
    row.update(overrides)
    return row


PASSING_KWARGS = dict(
    gnomad_af=0.05,
    vaf=(0.01, 0.25),
    coverage=500,
    read_pos=0.2,
    strand_bias=(0.25, 0.75),
    zscore=5.0,
    low_complexity=True,
    filters=["PASS"],
    gene_list=[],
    chrom_list=[],
)


def run(df, **overrides):
    kwargs = {**PASSING_KWARGS, **overrides}
    in_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    return filter_results(in_df, **kwargs)


def test_row_passing_every_filter_is_kept():
    result = run([make_row()])
    assert len(result) == 1


def test_variant_not_called_is_excluded():
    result = run([make_row(variant_called=False)])
    assert result.empty


@pytest.mark.parametrize(
    "field,value",
    [
        ("gnomADw4.1_AF", 0.05),  # equal to threshold, .lt() is strict
        ("base_fraction_alt", 0.01),  # equal to vaf lower bound
        ("base_fraction_alt", 0.25),  # equal to vaf upper bound
        ("depth", 500),  # equal to coverage threshold
        ("avg_pos_as_fraction_delta", 0.2),  # equal to read_pos threshold
        ("plus_strand_bias_alt", 0.25),  # equal to strand_bias lower bound
        ("plus_strand_bias_alt", 0.75),  # equal to strand_bias upper bound
        ("zscore_alt", 5.0),  # equal to zscore threshold
    ],
)
def test_threshold_boundaries_are_exclusive(field, value):
    result = run([make_row(**{field: value})])
    assert result.empty


def test_low_complexity_filter_drops_soft_masked_when_enabled():
    result = run([make_row(soft_masked=True)], low_complexity=True)
    assert result.empty


def test_low_complexity_filter_keeps_soft_masked_when_disabled():
    result = run([make_row(soft_masked=True)], low_complexity=False)
    assert len(result) == 1


def test_caller_filter_keeps_rows_with_matching_status():
    result = run(
        [make_row(callers_filter_alt="PASS, LowQual")],
        filters=["LowQual"],
    )
    assert len(result) == 1


def test_caller_filter_drops_rows_with_no_matching_status():
    result = run(
        [make_row(callers_filter_alt="LowQual")],
        filters=["PASS"],
    )
    assert result.empty


def test_caller_filter_handles_nan_callers_filter():
    result = run([make_row(callers_filter_alt=float("nan"))], filters=["PASS"])
    assert result.empty


def test_empty_filters_list_drops_all_rows():
    # A row that would pass every other filter is still dropped once
    # `filters` is empty: `set(x) & set([])` is always empty, so nothing
    # can match. This mirrors the sidebar's default (no filter statuses
    # selected), so results start empty until the user picks at least one.
    result = run([make_row()], filters=[])
    assert result.empty


def test_gene_list_empty_does_not_filter():
    result = run([make_row(SYMBOL="TP53")], gene_list=[])
    assert len(result) == 1


def test_gene_list_filters_to_selected_genes():
    result = run(
        [make_row(SYMBOL="TP53"), make_row(SYMBOL="BRCA1")],
        gene_list=["BRCA1"],
    )
    assert result["SYMBOL"].tolist() == ["BRCA1"]


def test_chrom_list_empty_does_not_filter():
    result = run([make_row(variant="chr7:200:C>G")], chrom_list=[])
    assert len(result) == 1


def test_chrom_list_filters_to_selected_chromosomes():
    result = run(
        [make_row(variant="chr1:100:A>T"), make_row(variant="chr7:200:C>G")],
        chrom_list=["chr7"],
    )
    assert result["variant"].tolist() == ["chr7:200:C>G"]


def test_alt_depth_filter_defaults_to_permissive():
    # alt_depth defaults to 0, so a row with few supporting reads isn't
    # affected unless the caller explicitly sets a threshold.
    result = run([make_row(count_alt=1)])
    assert len(result) == 1


def test_alt_depth_filter_drops_rows_below_threshold():
    result = run([make_row(count_alt=5)], alt_depth=10)
    assert result.empty


def test_alt_depth_filter_keeps_rows_above_threshold():
    result = run([make_row(count_alt=15)], alt_depth=10)
    assert len(result) == 1


def test_alt_depth_filter_boundary_is_exclusive():
    result = run([make_row(count_alt=10)], alt_depth=10)
    assert result.empty


def test_all_rows_filtered_out_returns_empty_without_error():
    # Regression guard: the caller-filter step is skipped once the
    # dataframe is already empty (see the `if not filtered_df.empty` guard
    # in filtering.py), which previously masked a KeyError elsewhere when
    # filters zeroed out all variant rows.
    result = run([make_row(**{"gnomADw4.1_AF": 0.9})])
    assert result.empty


def test_empty_input_dataframe_returns_empty_without_error():
    columns = [
        "variant",
        "variant_called",
        "gnomADw4.1_AF",
        "base_fraction_alt",
        "depth",
        "count_alt",
        "avg_pos_as_fraction_delta",
        "plus_strand_bias_alt",
        "zscore_alt",
        "soft_masked",
        "callers_filter_alt",
        "SYMBOL",
    ]
    result = run(pd.DataFrame(columns=columns))
    assert result.empty


PT_INFO = pd.DataFrame(
    [
        {"Sequencing Name": "s1", "Sample Status": "Affected", "Sequencing Round": 1},
        {"Sequencing Name": "s2", "Sample Status": "Affected", "Sequencing Round": 2},
        {"Sequencing Name": "s3", "Sample Status": "Control", "Sequencing Round": 1},
    ]
)


def test_samples_matching_single_column():
    assert samples_matching(PT_INFO, {"Sample Status": ["Control"]}) == {"s3"}


def test_samples_matching_unions_across_columns():
    matched = samples_matching(
        PT_INFO, {"Sample Status": ["Control"], "Sequencing Round": [2]}
    )
    assert matched == {"s2", "s3"}


def test_samples_matching_ignores_columns_with_no_selection():
    assert samples_matching(PT_INFO, {"Sample Status": [], "Sequencing Round": []}) == set()


def test_samples_matching_value_absent_from_frame():
    assert samples_matching(PT_INFO, {"Sample Status": ["Unknown"]}) == set()


def test_build_export_csv_header_and_roundtrip():
    unfiltered = pd.DataFrame(
        [
            make_row(sample="s1"),
            make_row(sample="s2", variant="chr2:5:G>C"),
            make_row(sample="s3", variant="chr3:9:T>A", variant_called=False),
        ]
    )
    filtered = unfiltered.iloc[[0]]
    csv_str = build_export_csv(
        filtered,
        unfiltered,
        filters={"gnomad_af": 0.05, "genes": ["BRCA1"]},
        meta={
            "analysis_time": datetime(2026, 8, 16, 13, 40, 2),
            "export_time": datetime(2026, 8, 16, 14, 3, 11),
            "hostname": "testhost",
            "inputs": {"normalized_variant": "/data/variants.tsv"},
            "excluded_samples": ["bad1"],
        },
    )
    lines = csv_str.splitlines()
    header = [l for l in lines if l.startswith("#")]
    # every line before the column header is a comment
    assert lines[: len(header)] == header
    # pre-filter counts only "variant_called" samples but all variants;
    # post-filter counts come from the filtered frame
    assert "# pre-filter: 2 samples, 3 unique variants" in header
    assert "# post-filter: 1 samples, 1 unique variants" in header
    assert "# analysis run: 2026-08-16T13:40:02" in header
    assert "# exported: 2026-08-16T14:03:11" in header
    assert "# input: normalized_variant=/data/variants.tsv" in header
    assert "# excluded samples: bad1" in header
    assert "# filter: gnomad_af=0.05" in header
    assert "# filter: genes=['BRCA1']" in header
    # pandas round-trips the data by skipping the comment block
    roundtrip = pd.read_csv(io.StringIO(csv_str), comment="#")
    assert len(roundtrip) == len(filtered)
    assert set(filtered.columns) <= set(roundtrip.columns)


def test_build_export_csv_handles_missing_analysis_time_and_no_exclusions():
    df = pd.DataFrame([make_row(sample="s1")])
    csv_str = build_export_csv(
        df,
        df,
        filters={},
        meta={
            "analysis_time": None,
            "export_time": datetime(2026, 8, 16, 14, 3, 11),
            "hostname": "testhost",
            "inputs": {},
            "excluded_samples": [],
        },
    )
    assert "# analysis run: unknown" in csv_str
    assert "# excluded samples: none" in csv_str
