"""Pure dataframe transforms for the analysis pipeline (no Streamlit)."""

import numpy as np
import pandas as pd


def parse_variants(variants):
    """Parse "chrom:pos:REF>ALT" strings into bam-readcount coordinates."""
    parsed_variants_df = pd.DataFrame({"variant": sorted(variants)})
    loc = parsed_variants_df["variant"].str.split(":", n=2, expand=True)
    ref_alt = loc[2].str.split(">", n=1, expand=True)
    ref, alt = ref_alt[0], ref_alt[1]
    ref_len, alt_len = ref.str.len(), alt.str.len()

    if ((ref_len > 1) & (alt_len > 1)).any():
        raise ValueError("unhandled mnp.")

    # ref[0] == alt for a deletion, so bam-readcount's alt is "-" + the
    # deleted bases past the shared anchor, and pos shifts past that anchor.
    is_del = ref_len > alt_len
    # ie: C>CAG would mean ref=C, alt=+AG for bam-readcount's alt.
    is_ins = alt_len > ref_len

    parsed_variants_df["chrom"] = loc[0]
    pos = loc[1].astype(int)
    parsed_variants_df["pos"] = np.where(is_del, pos + 1, pos)
    parsed_variants_df["readcount_ref"] = np.where(is_del, ref.str[1], ref)
    parsed_variants_df["readcount_alt"] = np.where(
        is_del,
        "-" + ref.str.slice(1),
        np.where(is_ins, "+" + alt.str.slice(1), alt),
    )
    parsed_variants_df["variant_type"] = np.select(
        [is_del, is_ins], ["del", "ins"], default="snp"
    )
    return parsed_variants_df


def classify_readcount_type(matches):
    """Label each readcount row alt/ref/offsite by its observed base."""
    return np.select(
        [
            matches["base"] == matches["readcount_alt"],
            matches["base"] == matches["readcount_ref"],
        ],
        ["alt", "ref"],
        default="offsite",
    )


def add_metrics(df):
    # keep ref/alt rows only; offsite bases have no column in the wide pivot
    df = df[df["readcount_type"].isin(["ref", "alt"])]
    # pivot long readcount rows to one wide row per (sample, variant)
    # TODO change this to take all columns not in values/bam-readcount results
    wide = df.pivot_table(
        index=[
            "sample",
            "chrom",
            "pos",
            "ref",
            "variant",
            "variant_type",
            "depth",
            "variant_called",
        ],
        columns="readcount_type",
        values=[
            "count",
            "avg_mapping_quality",
            "avg_basequality",
            "avg_se_mapping_quality",
            "num_plus_strand",
            "avg_pos_as_fraction",
            "num_minus_strand",
            "avg_num_mismatches_as_fraction",
            "avg_sum_mismatch_qualities",
            "num_q2_containing_reads",
            "avg_distance_to_q2_start_in_q2_reads",
            "avg_clipped_length",
            "avg_distance_to_effective_3p_end",
            "base_fraction",
            "plus_strand_bias",
            "base",
            "callers",
            "callers_filter",
            "callers_count",
        ],
        aggfunc="first",
    ).reset_index()
    num_cols = wide.select_dtypes(include="number").columns
    wide[num_cols] = wide[num_cols].fillna(0)
    str_cols = wide.select_dtypes(include="str").columns
    wide[str_cols] = wide[str_cols].fillna("")
    wide["avg_pos_as_fraction_delta"] = abs(
        wide[("avg_pos_as_fraction", "ref")] - wide[("avg_pos_as_fraction", "alt")]
    )
    # z-score of alt base_fraction across samples, per variant
    wide[("zscore", "alt")] = (
        wide[("base_fraction", "alt")]
        .groupby(wide[("variant", "")])
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    wide.columns = ["_".join([i for i in col if i]) for col in wide.columns]
    return wide
