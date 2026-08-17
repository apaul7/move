from os import listdir
from os.path import isfile, join
import os
import re
import tomllib
import streamlit as st
import numpy as np
import pandas as pd
import time
import csv
import altair as alt
import pickle
import streamlit.components.v1 as components
from pathlib import Path
import socket
import argparse
import logging
import sys
import requests
from natsort import natsort_keygen

from analysis import add_metrics, classify_readcount_type, parse_variants
from datetime import datetime

from filtering import build_export_csv, filter_results, samples_matching
import gene_track


def parse_args():
    # streamlit passes script args through after a literal "--", e.g.
    # `streamlit run app.py -- --debug`; parse_known_args() ignores anything
    # else (e.g. streamlit's own flags) that ends up in sys.argv.
    parser = argparse.ArgumentParser(description="MoVE - Mosaic Variant Explorer")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug-level logging",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


_args = parse_args()

logging.basicConfig(
    level=logging.DEBUG if _args.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("move")
logger.debug("Debug logging enabled")

# Paths are environment-specific (differ per machine/deployment), so they live
# in a TOML config file rather than being hardcoded here. See
# config.example.toml for the expected format.
REQUIRED_PATH_KEYS = [
    "merged_readcount_dir",
    "normalized_variant_path",
    "raw_variant_path",
    "annotation_path",
    "igv_soft_mask_path",
    "patient_info_path",
]
DIRECTORY_PATH_KEYS = {"merged_readcount_dir"}

# Patient info columns pre-selected in the sidebar's "Filter by column(s)"
# picker. Matched case-insensitively; any column not present in the loaded
# excel is skipped, so this list can be extended freely without breaking excel
# files that lack a given column. Users can add any other column at runtime,
# so this is a starting point rather than a whitelist.
PATIENT_INFO_FILTER_COLUMNS = ["status", "sequencing_round"]

# Standard filtering presets for the "Filtering results" section. Mosaic is
# the app's baseline (low VAF, high z-score confidence); Germline relaxes
# the z-score floor and widens VAF to cover het/hom calls instead; X-linked
# drops the VAF ceiling only, since a hemizygous chrX locus isn't diluted by
# a second allele the way an autosomal mosaic is.
MOSAIC_FILTER_DEFAULTS = {
    "filter_lcr": True,
    "filter_gnomad_af": 0.05,
    "filter_vaf": (0.01, 0.25),
    "filter_coverage": 500,
    "filter_read_pos": 0.2,
    "filter_strand_bias": (0.25, 0.75),
    "filter_zscore": 5.0,
    "filter_alt_depth": 0,
}
GERMLINE_FILTER_DEFAULTS = {
    **MOSAIC_FILTER_DEFAULTS,
    "filter_zscore": 0.0,
    "filter_vaf": (0.25, 1.0),
}
X_LINKED_FILTER_DEFAULTS = {
    **MOSAIC_FILTER_DEFAULTS,
    "filter_vaf": (0.01, 1.0),
}


def load_paths_config():
    config_path = os.environ.get("MOVE_CONFIG_PATH", "config.toml")
    logger.debug("Loading paths config from %s", config_path)

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        st.error(
            f"Config file not found: {config_path}\n\n"
            "Set the MOVE_CONFIG_PATH environment variable or create "
            "config.toml (see config.example.toml for the expected format)."
        )
        st.stop()
    except tomllib.TOMLDecodeError as e:
        logger.error("Failed to parse config file %s: %s", config_path, e)
        st.error(f"Failed to parse config file {config_path}: {e}")
        st.stop()

    paths = config.get("paths", {})
    missing_keys = [key for key in REQUIRED_PATH_KEYS if key not in paths]
    if missing_keys:
        logger.error(
            "Config file %s missing required [paths] keys: %s",
            config_path,
            ", ".join(missing_keys),
        )
        st.error(
            f"Config file {config_path} is missing required [paths] keys: "
            + ", ".join(missing_keys)
        )
        st.stop()

    missing_paths = []
    for key in REQUIRED_PATH_KEYS:
        value = paths[key]
        if key in DIRECTORY_PATH_KEYS:
            exists, kind = os.path.isdir(value), "directory"
        else:
            exists, kind = os.path.isfile(value), "file"
        if not exists:
            missing_paths.append(f"{key} ({kind} not found): {value}")
    if missing_paths:
        logger.error(
            "Config file %s points to paths that don't exist: %s",
            config_path,
            "; ".join(missing_paths),
        )
        st.error(
            f"Config file {config_path} points to paths that don't exist:\n\n"
            + "\n".join(f"- {p}" for p in missing_paths)
        )
        st.stop()

    logger.debug("Paths config loaded: %s", paths)
    return paths


def public_host() -> str:
    # Host header as sent by the browser, e.g. "c2-node-095.ris.wustl.edu:8590".
    # The go proxy intentionally does not rewrite Host (see go/main.go), so this
    # is the address the user is actually connected through -- right node, right
    # port, no hardcoding. socket.gethostname() would return the node's short
    # name, which is not resolvable from off the cluster. Falls back to it for
    # bare `streamlit run` outside the proxy.
    return st.context.headers.get("Host") or socket.gethostname()

st.set_page_config(
    page_title="MoVE",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_paths = load_paths_config()
merged_readcount_dir = _paths["merged_readcount_dir"]
normalized_variant_path = _paths["normalized_variant_path"]
raw_variant_path = _paths["raw_variant_path"]
annotation_path = _paths["annotation_path"]
igv_soft_mask_path = _paths["igv_soft_mask_path"]
patient_info_path = _paths["patient_info_path"]


@st.cache_data
def load_tsv(tsv_path, header=True):
    logger.debug("Loading TSV: %s", tsv_path)
    pd_header = "infer" if header else None
    try:
        df = pd.read_csv(tsv_path, sep="\t", header=pd_header)
    except FileNotFoundError as e:
        logger.error("File not found: %s", tsv_path)
        st.error(f"File not found: {tsv_path}")
        raise e
    except Exception as e:
        logger.error("Failed to load TSV file: %s w/ error: %s", tsv_path, e)
        st.error(f"Failed to load TSV file: {tsv_path} w/ error:\n{e}")
        raise e
    logger.debug("Loaded TSV %s with shape %s", tsv_path, df.shape)
    return df


@st.cache_data
def load_excel(excel_path, sheet):
    logger.debug("Loading excel: %s (sheet=%s)", excel_path, sheet)
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet)
    except FileNotFoundError as e:
        logger.error("File not found: %s", excel_path)
        st.error(f"File not found: {excel_path}")
        raise e
    except Exception as e:
        logger.error("Failed to load excel file: %s w/ error: %s", excel_path, e)
        st.error(f"Failed to load excel file: {excel_path} w/ error:\n{e}")
        raise e
    logger.debug("Loaded excel %s with shape %s", excel_path, df.shape)
    return df


EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
TSV_EXTENSIONS = {".tsv", ".txt"}


def load_patient_info(path, sheet="Patient Information"):
    # Headers aren't hardcoded here: whatever columns the file has are
    # inferred by pandas and carried through, so patient info files can
    # freely gain/lose columns without app.py changes.
    ext = Path(path).suffix.lower()
    if ext in EXCEL_EXTENSIONS:
        return load_excel(excel_path=path, sheet=sheet)
    if ext in TSV_EXTENSIONS:
        return load_tsv(tsv_path=path)
    st.error(
        f"Unsupported patient info file type '{ext}': {path}\n\n"
        f"Expected one of: {', '.join(sorted(EXCEL_EXTENSIONS | TSV_EXTENSIONS))}"
    )
    st.stop()


@st.cache_data
def cached_gene_region(gene_symbol):
    logger.debug("Resolving UCSC hg38 region for gene %s", gene_symbol)
    region = gene_track.search_gene_region(gene_symbol)
    if region is None:
        logger.error("No hg38 region found on UCSC for gene %s", gene_symbol)
    else:
        logger.debug("Resolved %s to %s", gene_symbol, region)
    return region


@st.cache_data
def cached_gene_transcripts(gene_symbol, chrom, start, end):
    logger.debug(
        "Fetching UCSC transcripts for %s (%s:%d-%d)", gene_symbol, chrom, start, end
    )
    transcripts = gene_track.fetch_transcripts(gene_symbol, chrom, start, end)
    logger.debug("Fetched %d transcript(s) for %s", len(transcripts), gene_symbol)
    return transcripts


@st.cache_data
def load_sample_names(path):
    df = load_tsv(path)
    sample_names = df["sample"].dropna().unique()
    logger.debug("Loaded %d sample names from %s", len(sample_names), path)
    return sample_names


@st.cache_data
def load_raw_to_normalized_variant_dict(path):
    variant_list = {}
    with open(path) as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            variant_list[row["og_vcf_key"]] = row["variant"]
    return variant_list


@st.cache_data
def load_raw_variants(path):
    df = load_tsv(path)
    df["unique_count"] = df.groupby("variant")["sample"].transform("nunique")
    variant_list = {variant: group for variant, group in df.groupby("variant")}
    return df


def filter_sample_variants(sample, raw_variants_df):
    return raw_variants_df[raw_variants_df["sample"] == sample]


@st.cache_data
def load_softmasked(path):
    df = load_tsv(path, header=False)
    df.columns = ["CHROM", "START", "STOP"]
    return df


@st.cache_data
def load_annotations(path):
    df = load_tsv(path)
    df["variant"] = (
        df["CHROM"] + ":" + df["POS"].astype(str) + ":" + df["REF"] + ">" + df["ALT"]
    )
    soft_masked_df = load_softmasked(igv_soft_mask_path)

    ref_len = df["REF"].str.len()
    alt_len = df["ALT"].str.len()
    df["END"] = df["POS"] + np.maximum(ref_len, alt_len) - 1

    df["soft_masked"] = False
    for chrom, sm in soft_masked_df.groupby("CHROM"):
        mask = df["CHROM"].eq(chrom)
        if not mask.any():
            continue
        ann = df.loc[mask]
        overlap = (ann["POS"].to_numpy()[:, None] >= sm["START"].to_numpy()) & (
            ann["END"].to_numpy()[:, None] <= sm["STOP"].to_numpy()
        )
        df.loc[mask, "soft_masked"] = overlap.any(axis=1)
    df = df.drop(["CHROM", "POS", "REF", "ALT", "END"], axis=1)
    return df


@st.cache_data
def merge_annotation_to_df(df, annotations_df, og_to_norm_dict, pt_info):
    # add og_to_norm() to df
    df["norm_variant"] = df["variant"].map(og_to_norm_dict)
    df["annotation_merge_key"] = df["norm_variant"].fillna(df["variant"])
    annotations_indexed = annotations_df.set_index("variant")
    pt_info_indexed = pt_info.set_index("Sequencing Name")
    merged_df = df.join(annotations_indexed, on="annotation_merge_key", how="left")
    merged_df = merged_df.join(pt_info_indexed, on="sample", how="left")
    return merged_df


@st.cache_data
def make_pie_chart(df, value):
    counts = df[value].value_counts().rename_axis("Category").reset_index(name="Count")
    plot = (
        alt.Chart(counts)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal"),
        )
    )
    return plot


@st.cache_data
def load_parquet(parquet_path, filters=None):
    logger.debug("Loading Parquet: %s (filtered=%s)", parquet_path, bool(filters))
    try:
        # filters are pushed down to pyarrow, so non-matching rows are never
        # materialized as pandas objects -- see add_readcount_to_df.
        df = pd.read_parquet(parquet_path, filters=filters)
    except FileNotFoundError as e:
        logger.error("File not found: %s", parquet_path)
        st.error(f"File not found: {parquet_path}")
        raise e
    except Exception as e:
        logger.error("Failed to load Parquet file: %s w/ error: %s", parquet_path, e)
        st.error(f"Failed to load Parquet file: {parquet_path} w/ error:\n{e}")
        raise e
    logger.debug("Loaded Parquet %s with shape %s", parquet_path, df.shape)
    return df


# not cached: load_parquet/load_tsv already cache, and a second decorator here
# would keep a second pickled copy of every readcount table in memory.
def load_sample_readcount(sample, merged_readcount_path, filters=None):
    # TODO: clarify -- should this fall back to the "norm" sample?
    parquet_path = Path(merged_readcount_path) / f"{sample}_bam_readcount.parquet"
    if parquet_path.exists():
        return load_parquet(str(parquet_path), filters)
    # TODO the TSV fallback has no pushdown, so it still reads the whole table.
    return load_tsv(merged_readcount_path + f"/{sample}_bam_readcount.tsv")


def add_readcount_to_df(df, readcount_path, samples):
    variants = df["variant"].unique()
    raw_variants = st.session_state.raw_variants
    called_pairs = set(zip(raw_variants["variant"], raw_variants["sample"]))
    sample_called_variants = set(
        raw_variants.loc[raw_variants["sample"].isin(samples), "variant"]
    )
    variants = set(variants) & sample_called_variants

    if not variants:
        logger.error(
            "No overlap between normalized VCF variants and raw-called "
            "variants for the selected sample(s): %s",
            samples,
        )
        st.error(
            "No variants found for the selected sample(s): "
            f"{', '.join(samples)}.\n\n"
            "Check that these sample names match the 'sample' column in "
            "the raw variant file, and that the normalized variant file "
            "contains variants called for them."
        )
        raise ValueError(f"No variants found for sample(s): {samples}")

    parsed_variants_df = parse_variants(variants)

    # A per-sample readcount table covers every base at every covered position
    # in the panel, but the join below keeps only the target variant positions.
    # Pushing that down to the parquet reader means the discarded rows are never
    # read into memory -- without it, peak usage is (samples x whole table).
    # This is a superset filter (chrom X pos, not exact pairs); the join narrows
    # it to the real pairs, and the extra rows are a rounding error.
    readcount_filters = [
        ("chrom", "in", sorted(parsed_variants_df["chrom"].unique().tolist())),
        ("pos", "in", sorted(parsed_variants_df["pos"].unique().tolist())),
    ]

    all_matches = []
    for i, sample in enumerate(samples):
        logger.debug("Processing readcount %d/%d: %s", i + 1, len(samples), sample)
        # long-format bam-readcount table: one row per (chrom, pos, base)
        sample_readcount_df = load_sample_readcount(
            sample, readcount_path, readcount_filters
        )
        matches = sample_readcount_df.merge(
            parsed_variants_df,
            on=["chrom", "pos"],
            how="inner",
        )
        matches["variant_called"] = [
            (v, sample) in called_pairs for v in matches["variant"]
        ]
        matches["base_fraction"] = matches["count"] / matches["depth"]
        matches["plus_strand_bias"] = matches["num_plus_strand"] / matches["count"]
        matches["readcount_type"] = classify_readcount_type(matches)
        all_matches.append(matches)
    results = pd.concat(all_matches, ignore_index=True)
    return results


def add_sample_call_status(df):
    # df has variant and sample.
    # process raw sample vcf to add caller count and filter status to df
    metrics = (
        st.session_state.raw_variants.groupby(["variant", "sample"])
        .agg(
            callers=("caller", lambda s: ", ".join(s.astype(str))),
            callers_filter=("FILTER", lambda s: ", ".join(s.astype(str))),
            callers_count=("caller", "size"),
        )
        .reset_index()
    )
    df = df.merge(
        metrics,
        on=["variant", "sample"],
        how="left",
    )
    return df


@st.cache_data
def run_analysis(vcf_path, readcount_path, samples, host):
    logger.info("Running analysis for %d sample(s)", len(samples))
    df = load_tsv(vcf_path)
    logger.debug("1) loaded vcf, shape=%s", df.shape)
    df = add_readcount_to_df(df, readcount_path, samples)
    logger.debug("2) added readcount, shape=%s", df.shape)
    df = add_sample_call_status(df)
    logger.debug("3) added sample call status, shape=%s", df.shape)
    df = add_metrics(df)
    logger.debug("4) added metrics, shape=%s", df.shape)
    df = merge_annotation_to_df(
        df,
        st.session_state.annotations,
        st.session_state.raw_to_norm_dict,
        st.session_state.pt_info,
    ).reset_index()
    logger.debug("5) merged annotations, shape=%s", df.shape)
    # add url
    df["igv"] = df.apply(
        # host already carries the port (it is the browser's Host header), so
        # none is appended here -- launching on a different PORT just works.
        lambda row: f"https://{host}/igvjs/?chromosome={row['variant'].split(':')[0]}&position={row['variant'].split(':')[1]}&sample={row['sample']}",
        axis=1,
    )
    front_cols = [
        "igv",
        "sample",
        "SYMBOL",
        "Consequence",
        "variant",
        "base_fraction_alt",
        "zscore_alt",
        "depth",
        "count_alt",
        "count_ref",
        "CADD_PHRED",
        "gnomADw4.1_AF",
        "gnomADw4.1_AF_grpmax",
        "HGVSc",
        "HGVSp",
        "callers_alt",
        "callers_filter_alt",
        "callers_count_alt",
        "soft_masked",
    ]
    df = df[front_cols + [c for c in df.columns if c not in front_cols]]
    df = df.drop(
        columns=[
            "callers_count_ref",
            "callers_filter_ref",
            "callers_ref",
            "chrom",
            "pos",
            "ref",
            "zscore_ref",
            "level_0",
            "norm_variant",
            "annotation_merge_key",
        ],
        errors="ignore",
    )
    logger.info("Analysis complete, final shape=%s", df.shape)
    return df


@st.cache_data
def make_scatter_plot(df, x_val, y_val, extras):
    plot = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(x=x_val, y=y_val, tooltip=extras)
        .interactive()
    )
    return plot


@st.cache_data
def make_dot_plot(df, x_val="variant", y_val="base_fraction_alt", tooltip=None):
    base = df.copy().dropna(subset=[y_val])
    jitter = (
        alt.Chart(base)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(f"{x_val}:N"),
            y=alt.Y(f"{y_val}:Q"),
            size=alt.Size("depth:Q", legend=alt.Legend(title="Depth")),
            tooltip=tooltip,
            xOffset=alt.XOffset("jitter:Q"),
            color=alt.condition(
                alt.datum.variant_called == True,
                alt.value("red"),
                alt.value("lightgray"),
            ),
        )
        .transform_calculate(jitter="(random() - 0.5) * 0.3")
    )
    median_tick = (
        alt.Chart(base)
        .mark_tick(color="black", thickness=2, size=30)
        .encode(
            x=alt.X(f"{x_val}:N"),
            y=alt.Y(f"median({y_val}):Q"),
        )
    )
    return (jitter + median_tick).interactive()


def _aggregate_variants_by_position(df, position_series):
    work = df.assign(_position=position_series)
    work = work.dropna(subset=["_position"])
    if work.empty:
        return work
    work["_position"] = work["_position"].astype(int)
    work["_protein_position"] = work["Protein_position"].apply(
        gene_track.parse_protein_position
    )
    return work.groupby(["_position", "Consequence"], as_index=False).agg(
        count=("sample", "nunique"),
        depth=("depth", "mean"),
        samples=("sample", lambda s: ", ".join(sorted(set(s)))),
        hgvsp=("HGVSp", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
        protein_positions=(
            "_protein_position",
            lambda s: ", ".join(str(p) for p in sorted(set(s.dropna().astype(int)))),
        ),
    )


def _variant_exon_number(disp_x, exon_boxes):
    for box in exon_boxes:
        if box["disp_start"] <= disp_x <= box["disp_end"]:
            return box["exon_number"]
    return None  # falls in an intron -- no exon to highlight


def make_gene_exon_plot(exon_boxes, gene_df, genomic_to_display):
    region_heights = {"cds": (-0.4, 0.4), "utr5": (-0.15, 0.15), "utr3": (-0.15, 0.15)}
    exon_df = pd.DataFrame(exon_boxes)
    exon_df["y0"] = exon_df["region"].map(lambda r: region_heights[r][0])
    exon_df["y1"] = exon_df["region"].map(lambda r: region_heights[r][1])

    # Clicking a variant in the track above highlights the exon it falls in,
    # linking the two panels instead of leaving them as unrelated charts.
    exon_click = alt.selection_point(
        fields=["exon_number"], on="click", empty=False, name="exon_highlight"
    )

    backbone = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [exon_df["disp_start"].min()],
                    "x2": [exon_df["disp_end"].max()],
                    "y": [0.0],
                }
            )
        )
        .mark_rule(color="#888888", strokeWidth=1)
        .encode(x=alt.X("x:Q", axis=None), x2="x2:Q", y=alt.Y("y:Q", axis=None))
    )
    boxes = (
        alt.Chart(exon_df)
        .mark_rect()
        .encode(
            x=alt.X("disp_start:Q", axis=None),
            x2="disp_end:Q",
            y=alt.Y("y0:Q", axis=None),
            y2="y1:Q",
            color=alt.condition(
                exon_click,
                alt.value("#ffb703"),
                alt.Color(
                    "region:N",
                    scale=alt.Scale(
                        domain=["utr5", "cds", "utr3"],
                        range=["#a3c9e2", "#2c6ea6", "#a3c9e2"],
                    ),
                    legend=alt.Legend(title="Region"),
                ),
            ),
            tooltip=["exon_number", "region"],
        )
    )
    gene_diagram = (backbone + boxes).properties(width="container", height=80)

    positions = gene_df["variant"].str.split(":").str[1].astype(int)
    grouped = _aggregate_variants_by_position(gene_df, positions)
    if grouped.empty:
        return gene_diagram.interactive()
    grouped = grouped.rename(columns={"_position": "genomic_position"})
    grouped["disp_x"] = grouped["genomic_position"].apply(genomic_to_display)
    grouped["exon_number"] = grouped["disp_x"].apply(
        lambda x: _variant_exon_number(x, exon_boxes)
    )
    grouped["zero"] = 0

    stems = (
        alt.Chart(grouped)
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("disp_x:Q", axis=None),
            y=alt.Y("count:Q", title="Samples with variant"),
            y2="zero:Q",
            color=alt.Color("Consequence:N"),
        )
    )
    heads = (
        alt.Chart(grouped)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("disp_x:Q", axis=None),
            y=alt.Y("count:Q"),
            size=alt.Size("depth:Q", legend=alt.Legend(title="Mean Depth")),
            color=alt.Color("Consequence:N"),
            tooltip=[
                "genomic_position",
                "protein_positions",
                "count",
                "depth",
                "Consequence",
                "hgvsp",
                "samples",
            ],
        )
        .add_params(exon_click)
    )
    variant_track = (stems + heads).properties(width="container", height=250)

    return alt.vconcat(variant_track, gene_diagram).resolve_scale(
        x="shared", color="independent"
    )


if "samples" not in st.session_state:
    st.session_state.samples = load_sample_names(raw_variant_path)
if "raw_to_norm_dict" not in st.session_state:
    st.session_state.raw_to_norm_dict = load_raw_to_normalized_variant_dict(
        normalized_variant_path
    )
if "raw_variants" not in st.session_state:
    st.session_state.raw_variants = load_raw_variants(raw_variant_path)
if "annotations" not in st.session_state:
    st.session_state.annotations = load_annotations(annotation_path)
if "sample_variants" not in st.session_state:
    st.session_state.sample_variants = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()
if "filter_results" not in st.session_state:
    st.session_state.filter_results = pd.DataFrame()
if "pt_info" not in st.session_state:
    st.session_state.pt_info = load_patient_info(patient_info_path)
if "filter_gene_list" not in st.session_state:
    st.session_state.filter_gene_list = []
if "filter_chrom_list" not in st.session_state:
    st.session_state.filter_chrom_list = []
if "sample_excluder" not in st.session_state:
    st.session_state.sample_excluder = []


def _merge_into_excluder(new_samples):
    current = set(st.session_state.sample_excluder) | set(new_samples)
    # keep list ordered/deduped consistently with the known sample order
    st.session_state.sample_excluder = [
        s for s in st.session_state.samples if s in current
    ]


def _split_pasted_samples(raw_text):
    tokens = [t.strip() for t in re.split(r"[,\s]+", raw_text) if t.strip()]
    known = set(st.session_state.samples)
    matched = [t for t in tokens if t in known]
    unmatched = [t for t in tokens if t not in known]
    return matched, unmatched


def _apply_pasted_samples():
    matched, unmatched = _split_pasted_samples(
        st.session_state.get("sample_paste_input", "")
    )
    if matched:
        _merge_into_excluder(matched)
    st.session_state.sample_paste_unmatched = unmatched


def _normalize_col_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_column(columns, wanted):
    """First column whose normalized name contains `wanted`, else None."""
    target = _normalize_col_name(wanted)
    return next((c for c in columns if target in _normalize_col_name(c)), None)


def _apply_filter_preset(preset):
    st.session_state.update(preset)


def _available_chroms(df):
    if df.empty:
        return []
    return (
        df["variant"]
        .str.split(":")
        .str[0]
        .sort_values(key=natsort_keygen())
        .unique()
        .tolist()
    )


def _apply_x_linked_preset():
    st.session_state.update(X_LINKED_FILTER_DEFAULTS)
    st.session_state.filter_chrom_list = (
        ["chrX"]
        if "chrX" in _available_chroms(st.session_state.analysis_results)
        else []
    )


def _available_filter_statuses(df):
    if df.empty:
        return []
    return (
        df["callers_filter_alt"]
        .dropna()
        .str.split(", ")
        .explode()
        .str.strip()
        .loc[lambda x: x.ne("")]
        .sort_values(key=natsort_keygen())
        .unique()
        .tolist()
    )


st.title(":microscope: MoVE")
st.caption("**Mosaic Variant Explorer** — explore mosaic variants found in genomic data.")
st.caption(":point_left: Use the sidebar to exclude sample(s) from the views below.")

# filtering
st.sidebar.header(":no_entry_sign: Exclude Samples")

with st.sidebar.expander("Paste sample list to exclude"):
    st.text_area(
        "Sample names (comma, space, or newline separated)",
        key="sample_paste_input",
        height=100,
    )
    st.button("Add to exclude list", on_click=_apply_pasted_samples)
    if st.session_state.get("sample_paste_unmatched"):
        st.warning(
            "Not found in samples: "
            + ", ".join(st.session_state.sample_paste_unmatched)
        )

_pt_info_in_samples = st.session_state.pt_info[
    st.session_state.pt_info["Sequencing Name"].isin(st.session_state.samples)
]

# Seed the column picker with whichever defaults this patient info file has.
# Only on first run: after that the widget owns the value.
if "filter_columns" not in st.session_state:
    st.session_state.filter_columns = [
        c
        for c in (
            _resolve_column(st.session_state.pt_info.columns, _col)
            for _col in PATIENT_INFO_FILTER_COLUMNS
        )
        if c is not None
    ]

st.sidebar.multiselect(
    "Filter by column(s)",
    # "Sequencing Name" is omitted: the sample excluder below already is that filter.
    [c for c in st.session_state.pt_info.columns if c != "Sequencing Name"],
    key="filter_columns",
)

for _col in st.session_state.filter_columns:
    _values = sorted(
        _pt_info_in_samples[_col].dropna().unique(), key=natsort_keygen()
    )
    st.sidebar.multiselect(
        f"Exclude by {_col}",
        _values,
        key=f"sample_filter_{_col}",
        format_func=str,
    )

st.sidebar.multiselect(
    "Hide sample(s) from all views below",
    st.session_state.samples,
    key="sample_excluder",
)

# Column-driven exclusions are derived fresh each run, so deselecting a value
# (or dropping the column) brings those samples straight back. The manual
# excluder above and the paste box stay additive and independent of it.
_hidden_samples = set(st.session_state.sample_excluder) | samples_matching(
    st.session_state.pt_info,
    {
        _col: st.session_state.get(f"sample_filter_{_col}", [])
        for _col in st.session_state.filter_columns
    },
)
if _hidden_samples:
    st.sidebar.caption(f"{len(_hidden_samples)} sample(s) hidden from views below")

# init tabs
raw, analysis, by_variant, gene_view = st.tabs(
    ["Raw Variants", "Analysis+Filtering", "By Variant", "Gene View"]
)
with raw:
    st.multiselect(
        "Review Sample(s)",
        [s for s in st.session_state.samples if s not in _hidden_samples],
        key="sample_selector",
    )
    # filter variants to query?
    # plot averages of selected samples?
    for sample in st.session_state.sample_selector:
        with st.expander(sample + " Raw Variants"):
            s = filter_sample_variants(sample, st.session_state.raw_variants)
            s = merge_annotation_to_df(
                s,
                st.session_state.annotations,
                st.session_state.raw_to_norm_dict,
                st.session_state.pt_info,
            )
            column_order = [
                "variant",
                "SYMBOL",
                "Consequence",
                "CADD_PHRED",
                "gnomADw4.1_AF",
                "gnomADw4.1_AF_grpmax",
                "HGVSc",
                "HGVSp",
            ]
            s = s[column_order + [col for col in s.columns if col not in column_order]]
            with st.container(horizontal=True):
                st.metric(
                    "Total Variants",
                    value=s["variant"].nunique(),
                    help="Total unique variants called",
                    format="compact",
                )
                st.metric(
                    "Unique Variants",
                    value=s[s["unique_count"] == 1]["variant"].nunique(),
                    help="Total variants found only in this sample(n=1)",
                    format="compact",
                )
                for caller in sorted(s["caller"].dropna().unique()):
                    st.metric(
                        caller,
                        value=s[s["caller"] == caller]["variant"].nunique(),
                        help=f"Total variants called by {caller}",
                        format="compact",
                    )
                st.metric(
                    "SNPS",
                    value=s[s["TYPE"] == "SNP"]["variant"].nunique(),
                    help="Total SNP variants",
                    format="compact",
                )
                st.metric(
                    "INDELS",
                    value=s[s["TYPE"] == "INDEL"]["variant"].nunique(),
                    help="Total INDEL variants",
                    format="compact",
                )
                st.metric(
                    "MNP",
                    value=s[s["TYPE"] == "MNP"]["variant"].nunique(),
                    help="Total multi nucleotide variants",
                    format="compact",
                )
            st.dataframe(
                s,
                hide_index=True,
                column_config={
                    "gnomADw4.1_AF": st.column_config.NumberColumn(format="percent"),
                    "gnomADw4.1_AF_grpmax": st.column_config.NumberColumn(
                        format="percent"
                    ),
                },
            )
            with st.container(horizontal=True):
                st.altair_chart(make_pie_chart(s, "Consequence"))
                st.altair_chart(make_pie_chart(s, "SYMBOL"))

with analysis:
    with st.expander("Sample Selection"):
        with st.form(key="analysis_form", border=False):
            st.multiselect(
                "Select samples for analysis",
                [s for s in st.session_state.samples if s not in _hidden_samples],
                key="sample_selector_analysis",
            )
            analysis_submit = st.form_submit_button(
                label="Load Samples", type="primary"
            )
            if analysis_submit:
                # captured outside the cached run_analysis so it reflects when
                # Load Samples was pressed, not when the cache was first filled
                st.session_state.analysis_time = datetime.now()
                st.session_state.analysis_results = run_analysis(
                    normalized_variant_path,
                    merged_readcount_dir,
                    st.session_state.sample_selector_analysis,
                    public_host(),
                )
                # default the filter-status multiselect to "select all" for
                # whatever statuses are present in this run's results
                st.session_state.filter_filter_status = _available_filter_statuses(
                    st.session_state.analysis_results
                )
            if not (st.session_state.analysis_results.empty):
                with st.container(horizontal=True):
                    st.metric(
                        "initial variants",
                        value=st.session_state.analysis_results["variant"].nunique(),
                        help="unique variants in analysis sample",
                        format="localized",
                    )
                    st.metric(
                        "unique samples",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_called"] == True
                        ]["sample"]
                        .dropna()
                        .nunique(),
                        help="unique sample count",
                        format="localized",
                    )
                    st.metric(
                        "unique genes",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_called"] == True
                        ]["SYMBOL"]
                        .dropna()
                        .nunique(),
                        help="unique gene count",
                        format="localized",
                    )
                    st.metric(
                        "Consequence",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_called"] == True
                        ]["Consequence"]
                        .dropna()
                        .nunique(),
                        help="unique VEP Consequence count",
                        format="localized",
                    )
                    st.metric(
                        "snps",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_type"] == "snp"
                        ]["variant"].nunique(),
                        help="Total snp variants",
                        format="localized",
                    )
                    st.metric(
                        "insertions",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_type"] == "ins"
                        ]["variant"].nunique(),
                        help="Total insertion variants",
                        format="localized",
                    )
                    st.metric(
                        "deletions",
                        value=st.session_state.analysis_results[
                            st.session_state.analysis_results["variant_type"] == "del"
                        ]["variant"].nunique(),
                        help="Total deletion variants",
                        format="localized",
                    )
                with st.container(horizontal=True):
                    st.altair_chart(
                        make_pie_chart(
                            st.session_state.analysis_results[
                                st.session_state.analysis_results["variant_called"]
                                == True
                            ],
                            "SYMBOL",
                        )
                    )
                    st.altair_chart(
                        make_pie_chart(
                            st.session_state.analysis_results[
                                st.session_state.analysis_results["variant_called"]
                                == True
                            ],
                            "Consequence",
                        )
                    )
                st.container(horizontal=True)
                st.divider()
                st.title("clinvar_CLNSIG = Pathogenic/Likely_pathogenic")
                st.dataframe(
                    st.session_state.analysis_results[
                        (st.session_state.analysis_results["variant_called"] == True)
                        & (
                            st.session_state.analysis_results["clinvar_CLNSIG"].isin(
                                ["Pathogenic/Likely_pathogenic", "Pathogenic"]
                            )
                        )
                    ],
                    column_config={
                        "igv": st.column_config.LinkColumn(
                            "IGV", display_text="Open IGV"
                        )
                    },
                )
    # Isolated in a fragment so clicking a preset button or the
    # Submit button only reruns this section -- without this,
    # Streamlit reruns the ENTIRE script on every interaction,
    # including the per-sample loop in the Raw Variants tab above,
    # which is what made these buttons feel slow.
    @st.fragment
    def _render_filtering_section():
        with st.expander("Filtering results"):
            st.caption(
                "Optional: the presets below set the thresholds in the form to "
                "standard values for mosaic (low-VAF, high-confidence), "
                "germline (heterozygous/homozygous VAF, no z-score floor), or "
                "X-linked (chrX, no VAF ceiling) variant calls. Applying a "
                "preset is not required -- adjust any threshold directly if "
                "your data calls for different cutoffs."
            )
            with st.container(horizontal=True):
                st.button(
                    "Mosaic (default)",
                    on_click=_apply_filter_preset,
                    args=(MOSAIC_FILTER_DEFAULTS,),
                    help="reset filters below to the standard mosaic-variant defaults",
                )
                st.button(
                    "Germline",
                    on_click=_apply_filter_preset,
                    args=(GERMLINE_FILTER_DEFAULTS,),
                    help="reset filters below to standard germline-variant defaults "
                    "(no z-score floor, wider VAF range)",
                )
                st.button(
                    "X-linked",
                    on_click=_apply_x_linked_preset,
                    help="reset filters below for X-linked variants: keeps the "
                    "mosaic VAF floor but drops the ceiling, since a hemizygous "
                    "chrX locus isn't diluted by a second allele the way an "
                    "autosomal mosaic is. Also sets the chrom filter below to "
                    "chrX.",
                )
            with st.form(key="filtering_form", border=False):
                with st.container(horizontal=True):
                    with st.container(horizontal=False):
                        st.checkbox(
                            "Low Complexity Filter",
                            value=MOSAIC_FILTER_DEFAULTS["filter_lcr"],
                            key="filter_lcr",
                            help="filter out variants in a softmasked region",
                        )
                    st.slider(
                        "gnomAD Filter %",
                        0.0,
                        1.0,
                        value=MOSAIC_FILTER_DEFAULTS["filter_gnomad_af"],
                        step=0.01,
                        key="filter_gnomad_af",
                        help="filter out variants with gnomAD4.1_AF greater than X%",
                        format="percent",
                    )
                    st.slider(
                        "VAF Filter",
                        0.0,
                        1.0,
                        value=MOSAIC_FILTER_DEFAULTS["filter_vaf"],
                        step=0.01,
                        key="filter_vaf",
                        help="filter out variants with vaf less than X and greater than Y",
                    )
                    st.slider(
                        "Coverage Filter",
                        0,
                        5000,
                        value=MOSAIC_FILTER_DEFAULTS["filter_coverage"],
                        key="filter_coverage",
                        help="filter out variants with coverage less than X",
                    )
                    st.slider(
                        "Alt Read Depth Filter",
                        0,
                        500,
                        value=MOSAIC_FILTER_DEFAULTS["filter_alt_depth"],
                        key="filter_alt_depth",
                        help="filter out variants with fewer than X reads "
                        "supporting the alt allele. Distinct from VAF (a "
                        "fraction) and Coverage (total depth) -- this is the "
                        "raw alt-supporting read count.",
                    )
                    st.slider(
                        "Read Position Filter",
                        0.0,
                        1.0,
                        value=MOSAIC_FILTER_DEFAULTS["filter_read_pos"],
                        step=0.01,
                        key="filter_read_pos",
                        help="filter out variants with delta read position of a fraction greater than X",
                        format="percent",
                    )
                    st.slider(
                        "Strand Bias Filter",
                        0.0,
                        1.0,
                        value=MOSAIC_FILTER_DEFAULTS["filter_strand_bias"],
                        step=0.01,
                        key="filter_strand_bias",
                        help="filter out variants with strand bias less than X and greater than Y",
                    )
                    st.slider(
                        "Z-score Filter",
                        0.0,
                        20.0,
                        value=MOSAIC_FILTER_DEFAULTS["filter_zscore"],
                        step=0.01,
                        key="filter_zscore",
                        help="filter out variants with Zscore less than X",
                    )
                with st.expander("Optional Filters"):
                    st.multiselect(
                        "Filter gene",
                        options=(
                            [""]
                            if st.session_state.analysis_results.empty
                            else (
                                st.session_state.analysis_results["SYMBOL"]
                                .sort_values()
                                .unique()
                                .tolist()
                            )
                        ),
                        key="filter_gene_list",
                        help="filter for variants in X gene(s)",
                    )
                    st.multiselect(
                        "Filter chrom",
                        options=(
                            [""]
                            if st.session_state.analysis_results.empty
                            else _available_chroms(st.session_state.analysis_results)
                        ),
                        key="filter_chrom_list",
                        help="filter for variants on X chromosome(s)",
                    )
                with st.container(horizontal=True):
                    st.multiselect(
                        "Filter Status",
                        options=(
                            [""]
                            if st.session_state.analysis_results.empty
                            else _available_filter_statuses(
                                st.session_state.analysis_results
                            )
                        ),
                        key="filter_filter_status",
                        help="filter based on the raw FILTER status of variants",
                    )
                filter_submit = st.form_submit_button(label="Submit", type="primary")
                if filter_submit:
                    logger.info(
                        "Applying filters: gnomad_af=%s vaf=%s coverage=%s "
                        "read_pos=%s strand_bias=%s zscore=%s low_complexity=%s "
                        "filters=%s gene_list=%s chrom_list=%s alt_depth=%s",
                        st.session_state.filter_gnomad_af,
                        st.session_state.filter_vaf,
                        st.session_state.filter_coverage,
                        st.session_state.filter_read_pos,
                        st.session_state.filter_strand_bias,
                        st.session_state.filter_zscore,
                        st.session_state.filter_lcr,
                        st.session_state.filter_filter_status,
                        st.session_state.filter_gene_list,
                        st.session_state.filter_chrom_list,
                        st.session_state.filter_alt_depth,
                    )
                    st.session_state.filter_results = filter_results(
                        in_df=st.session_state.analysis_results,
                        gnomad_af=st.session_state.filter_gnomad_af,
                        vaf=st.session_state.filter_vaf,
                        coverage=st.session_state.filter_coverage,
                        read_pos=st.session_state.filter_read_pos,
                        strand_bias=st.session_state.filter_strand_bias,
                        zscore=st.session_state.filter_zscore,
                        low_complexity=st.session_state.filter_lcr,
                        filters=st.session_state.filter_filter_status,
                        gene_list=st.session_state.filter_gene_list,
                        chrom_list=st.session_state.filter_chrom_list,
                        alt_depth=st.session_state.filter_alt_depth,
                    )
                    logger.info(
                        "Filter results shape=%s", st.session_state.filter_results.shape
                    )
                    if st.session_state.filter_results.empty:
                        st.session_state.export_csv = ""
                        st.info("No variants passed the current filters.")
                    else:
                        st.dataframe(
                            st.session_state.filter_results.reset_index(),
                            column_config={
                                "igv": st.column_config.LinkColumn(
                                    "IGV", display_text="Open IGV"
                                )
                            },
                        )
                        # built here (at submit) so the header matches the
                        # filters actually applied; rendered after the form
                        # below, since st.download_button can't live in a form
                        st.session_state.export_csv = build_export_csv(
                            st.session_state.filter_results,
                            st.session_state.analysis_results,
                            filters={
                                "gnomad_af": st.session_state.filter_gnomad_af,
                                "vaf": st.session_state.filter_vaf,
                                "coverage": st.session_state.filter_coverage,
                                "alt_depth": st.session_state.filter_alt_depth,
                                "read_pos": st.session_state.filter_read_pos,
                                "strand_bias": st.session_state.filter_strand_bias,
                                "zscore": st.session_state.filter_zscore,
                                "exclude_low_complexity": st.session_state.filter_lcr,
                                "filter_status": st.session_state.filter_filter_status,
                                "genes": st.session_state.filter_gene_list,
                                "chroms": st.session_state.filter_chrom_list,
                            },
                            meta={
                                "analysis_time": st.session_state.get(
                                    "analysis_time"
                                ),
                                "export_time": datetime.now(),
                                "hostname": socket.gethostname(),
                                "inputs": {
                                    "normalized_variant": normalized_variant_path,
                                    "merged_readcount_dir": merged_readcount_dir,
                                    "annotation": annotation_path,
                                },
                                "excluded_samples": sorted(_hidden_samples),
                            },
                        )
                        with st.container(horizontal=True):
                            st.metric(
                                "initial variants",
                                value=st.session_state.filter_results["variant"]
                                .dropna()
                                .nunique(),
                                help="unique variants in analysis sample",
                            )
                            st.metric(
                                "unique samples",
                                value=st.session_state.filter_results["sample"]
                                .dropna()
                                .nunique(),
                                help="unique sample count",
                            )
                            st.metric(
                                "unique genes",
                                value=st.session_state.filter_results["SYMBOL"]
                                .dropna()
                                .nunique(),
                                help="unique gene count",
                            )
                            st.metric(
                                "Consequence",
                                value=st.session_state.filter_results[
                                    "Consequence"
                                ].nunique(),
                                help="unique gene count",
                            )
                            st.metric(
                                "snps",
                                value=st.session_state.filter_results[
                                    st.session_state.filter_results["variant_type"]
                                    == "snp"
                                ]["variant"].nunique(),
                                help="Total snp variants",
                                format="localized",
                            )
                            st.metric(
                                "insertions",
                                value=st.session_state.filter_results[
                                    st.session_state.filter_results["variant_type"]
                                    == "ins"
                                ]["variant"].nunique(),
                                help="Total insertion variants",
                                format="localized",
                            )
                            st.metric(
                                "deletions",
                                value=st.session_state.filter_results[
                                    st.session_state.filter_results["variant_type"]
                                    == "del"
                                ]["variant"].nunique(),
                                help="Total deletion variants",
                                format="localized",
                            )
                        with st.container(horizontal=True):
                            st.altair_chart(
                                make_scatter_plot(
                                    df=st.session_state.filter_results,
                                    x_val="sample",
                                    y_val="base_fraction_alt",
                                    extras=[
                                        "SYMBOL",
                                        "Consequence",
                                        "variant",
                                        "depth",
                                        "HGVSc",
                                        "HGVSp",
                                        "callers_filter_alt",
                                        "callers_alt",
                                    ],
                                )
                            )
                            st.altair_chart(
                                make_scatter_plot(
                                    df=st.session_state.filter_results,
                                    x_val="sample",
                                    y_val="zscore_alt",
                                    extras=[
                                        "SYMBOL",
                                        "Consequence",
                                        "variant",
                                        "depth",
                                        "HGVSc",
                                        "HGVSp",
                                        "callers_filter_alt",
                                        "callers_alt",
                                    ],
                                )
                            )
                        with st.container(horizontal=True):
                            st.altair_chart(
                                make_pie_chart(st.session_state.filter_results, "SYMBOL")
                            )
                            st.altair_chart(
                                make_pie_chart(
                                    st.session_state.filter_results, "Consequence"
                                )
                            )
            if st.session_state.get("export_csv"):
                st.download_button(
                    "Download CSV (with metadata)",
                    data=st.session_state.export_csv,
                    file_name=f"move_filtered_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                    # "ignore" stops the click rerunning this fragment, which
                    # would otherwise clear the results table above (it only
                    # renders on the Submit rerun)
                    on_click="ignore",
                )

    _render_filtering_section()

with by_variant:
    if not (st.session_state.analysis_results.empty):
        st.multiselect(
            "Review Variant",
            st.session_state.analysis_results["variant"].unique(),
            key="plot_variants",
        )
        st.text(st.session_state.plot_variants)
        st.altair_chart(
            make_dot_plot(
                df=st.session_state.analysis_results[
                    st.session_state.analysis_results["variant"].isin(
                        st.session_state.plot_variants
                    )
                ],
                x_val="variant",
                y_val="base_fraction_alt",
                tooltip=[
                    "sample",
                    "SYMBOL",
                    "Consequence",
                    "variant",
                    "depth",
                    "HGVSc",
                    "HGVSp",
                    "callers_filter_alt",
                    "callers_alt",
                ],
            )
        )
    else:
        st.text("load sample data in analysis tab")

with gene_view:
    if not st.session_state.analysis_results.empty:
        genes = (
            st.session_state.analysis_results["SYMBOL"]
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        if not genes:
            st.text("no genes available in the current analysis results")
        else:
            st.selectbox("Gene", genes, key="gene_view_symbol")
            selected_gene = st.session_state.gene_view_symbol
            gene_df = st.session_state.analysis_results[
                (st.session_state.analysis_results["SYMBOL"] == selected_gene)
                & (st.session_state.analysis_results["variant_called"] == True)
            ]
            if gene_df.empty:
                st.info(f"No called variants for {selected_gene}.")
            else:
                try:
                    with st.spinner(
                        f"Fetching hg38 gene structure for {selected_gene} from UCSC..."
                    ):
                        region = cached_gene_region(selected_gene)
                        if region is None:
                            st.warning(
                                f"No hg38 region found on UCSC for gene "
                                f"'{selected_gene}'."
                            )
                        else:
                            chrom, start, end = region
                            transcripts = cached_gene_transcripts(
                                selected_gene, chrom, start, end
                            )
                            preferred_transcript_id = None
                            if "Feature" in gene_df.columns:
                                features = gene_df["Feature"].dropna()
                                if not features.empty:
                                    preferred_transcript_id = features.iloc[0]
                            transcript = gene_track.select_transcript(
                                transcripts,
                                preferred_transcript_id=preferred_transcript_id,
                            )
                            if transcript is None:
                                st.warning(
                                    f"No UCSC transcript found for gene "
                                    f"'{selected_gene}' in {chrom}:{start}-{end}."
                                )
                            else:
                                if preferred_transcript_id and (
                                    gene_track.strip_transcript_version(
                                        transcript["name"]
                                    )
                                    != gene_track.strip_transcript_version(
                                        preferred_transcript_id
                                    )
                                ):
                                    st.caption(
                                        f"Showing UCSC transcript {transcript['name']} "
                                        f"(annotation used {preferred_transcript_id})"
                                    )
                                exon_boxes, genomic_to_display = (
                                    gene_track.build_display_layout(transcript)
                                )
                                st.altair_chart(
                                    make_gene_exon_plot(
                                        exon_boxes, gene_df, genomic_to_display
                                    ),
                                    # width="stretch" alone doesn't actually
                                    # stretch vconcat charts to the container in
                                    # this Streamlit version -- it configures the
                                    # spec's autosize correctly but never sets
                                    # the proto flag the frontend needs to inject
                                    # the container's pixel width. use_container_width
                                    # is deprecated but is the only thing that
                                    # currently sets that flag.
                                    use_container_width=True,
                                )
                except requests.exceptions.RequestException as e:
                    logger.error(
                        "UCSC request failed for gene %s: %s", selected_gene, e
                    )
                    st.warning(f"Could not reach UCSC to fetch gene structure: {e}")
    else:
        st.text("load sample data in analysis tab")
