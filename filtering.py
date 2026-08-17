import streamlit as st


def samples_matching(pt_info, column_selections, sample_key="Sequencing Name"):
    """Sequencing names whose patient info row matches any selected value.

    column_selections maps a patient info column name to the list of values
    picked for it in the sidebar; a column with an empty list contributes
    nothing. Columns are OR'd together, so a sample matching any selection in
    any column is returned.
    """
    matched = set()
    for column, values in column_selections.items():
        if values:
            matched |= set(pt_info.loc[pt_info[column].isin(values), sample_key])
    return matched


def build_export_csv(filtered_df, unfiltered_df, filters, meta):
    """Filtered results as CSV text, preceded by '#' comment lines recording
    provenance: run/export times, input paths, pre/post-filter counts, and the
    filter settings applied. Re-read with pd.read_csv(..., comment="#").
    """
    pre_called = unfiltered_df[unfiltered_df["variant_called"] == True]
    analysis_time = meta.get("analysis_time")
    lines = [
        "# MoVE filtered variant export",
        f"# exported: {meta['export_time']:%Y-%m-%dT%H:%M:%S}",
        "# analysis run: "
        + (f"{analysis_time:%Y-%m-%dT%H:%M:%S}" if analysis_time else "unknown"),
        f"# host: {meta['hostname']}",
        *(f"# input: {name}={path}" for name, path in meta["inputs"].items()),
        f"# excluded samples: {', '.join(meta['excluded_samples']) or 'none'}",
        f"# pre-filter: {pre_called['sample'].dropna().nunique()} samples, "
        f"{unfiltered_df['variant'].nunique()} unique variants",
        f"# post-filter: {filtered_df['sample'].dropna().nunique()} samples, "
        f"{filtered_df['variant'].dropna().nunique()} unique variants",
        *(f"# filter: {key}={value}" for key, value in filters.items()),
    ]
    return "\n".join(lines) + "\n" + filtered_df.reset_index().to_csv(index=False)


@st.cache_data()
def filter_results(
    in_df,
    gnomad_af,
    vaf,
    coverage,
    read_pos,
    strand_bias,
    zscore,
    low_complexity,
    filters,
    gene_list,
    chrom_list,
    alt_depth=0,
):
    filtered_df = in_df[in_df["variant_called"] == True]

    filtered_df = filtered_df[filtered_df["gnomADw4.1_AF"].lt(gnomad_af)]
    filtered_df = filtered_df[
        filtered_df["base_fraction_alt"].gt(vaf[0])
        & filtered_df["base_fraction_alt"].lt(vaf[1])
    ]
    filtered_df = filtered_df[filtered_df["depth"].gt(coverage)]
    filtered_df = filtered_df[filtered_df["count_alt"].gt(alt_depth)]

    filtered_df = filtered_df[filtered_df["avg_pos_as_fraction_delta"].lt(read_pos)]
    filtered_df = filtered_df[
        filtered_df["plus_strand_bias_alt"].gt(strand_bias[0])
        & filtered_df["plus_strand_bias_alt"].lt(strand_bias[1])
    ]
    filtered_df = filtered_df[filtered_df["zscore_alt"].gt(zscore)]
    if low_complexity:
        filtered_df = filtered_df[filtered_df["soft_masked"] == False]
    if not filtered_df.empty:
        filtered_df = filtered_df[
            filtered_df["callers_filter_alt"]
            .fillna("")
            .str.split(", ")
            .apply(lambda x: bool(set(x) & set(filters)))
        ]
    if gene_list:
        filtered_df = filtered_df[filtered_df["SYMBOL"].isin(gene_list)]
    if chrom_list:
        chroms = filtered_df["variant"].str.split(":").str[0]
        filtered_df = filtered_df[chroms.isin(chrom_list)]
    return filtered_df
