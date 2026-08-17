"""The readcount pushdown in add_readcount_to_df assumes pyarrow applies
`filters` per row, not just per row group. If that assumption breaks, the
analysis silently loses variant rows -- so pin it here.
"""

from pathlib import Path

import pandas as pd

READCOUNT = (
    Path(__file__).parent.parent / "test_data" / "readcount" / "s10_bam_readcount.parquet"
)


def test_pushdown_matches_full_read():
    full = pd.read_parquet(READCOUNT)
    chroms = sorted(full["chrom"].unique().tolist())[:1]
    positions = sorted(full["pos"].unique().tolist())[:5]

    pushed = pd.read_parquet(
        READCOUNT,
        filters=[("chrom", "in", chroms), ("pos", "in", positions)],
    )
    expected = full[full["chrom"].isin(chroms) & full["pos"].isin(positions)]

    assert len(pushed) < len(full), "filter matched everything; test proves nothing"
    pd.testing.assert_frame_equal(
        pushed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_categorical=False,
    )
