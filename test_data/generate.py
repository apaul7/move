"""
One-off generator for MoVE dummy test data (100 samples, each with its own
unique low-VAF/mosaic SNP), built on top of the real test_data/annotations.tsv
positions so the annotation merge in app.py actually resolves to real
gene/consequence data instead of blanks.

Not part of the app; run once to (re)populate test_data/.
"""
import os
import random
import pandas as pd

random.seed(42)

TEST_DATA = os.path.dirname(os.path.abspath(__file__))

CALLERS = ["deepsomatic", "gatk", "mutect", "varscan"]
FILTER_CHOICES = ["PASS", "PASS", "PASS", "MVF5", "MQD30", "PB10", "GERMLINE"]
N_SAMPLES = 100
N_BACKGROUND = 15

# --- pull the SNP catalog from the real annotations.tsv (CHROM,POS,REF,ALT are
# the first 4 columns; the file is headerless, see app.py mismatch noted separately) ---
ann = pd.read_csv(
    f"{TEST_DATA}/annotations.tsv",
    sep="\t",
    header=None,
    usecols=[0, 1, 2, 3],
    names=["CHROM", "POS", "REF", "ALT"],
    dtype=str,
)
snp = ann[(ann["REF"].str.len() == 1) & (ann["ALT"].str.len() == 1)]
snp = snp.drop_duplicates(subset=["CHROM", "POS", "REF", "ALT"]).reset_index(drop=True)

pool = list(snp.itertuples(index=False, name=None))  # (CHROM, POS, REF, ALT)
random.shuffle(pool)

background_variants = pool[:N_BACKGROUND]
unique_variants = pool[N_BACKGROUND : N_BACKGROUND + N_SAMPLES]
assert len(unique_variants) == N_SAMPLES

samples = [f"s{i+1}" for i in range(N_SAMPLES)]


def variant_str(v):
    chrom, pos, ref, alt = v
    return f"{chrom}:{pos}:{ref}>{alt}"


def make_gt_info(af, depth):
    alt_depth = round(depth * af)
    ref_depth = depth - alt_depth
    return f"0/1:{ref_depth},{alt_depth}:{af:.3f}:{depth}:0,0:0,0:{ref_depth},{alt_depth}:0,0,0,0"


# ---------- raw.sample_call_key.tsv ----------
raw_rows = []
# background: each recurs in a random subset of samples, called by 1-3 callers
for v in background_variants:
    vs = variant_str(v)
    n_carriers = random.randint(5, 60)
    carriers = random.sample(samples, n_carriers)
    for sample in carriers:
        n_callers = random.randint(1, 3)
        for caller in random.sample(CALLERS, n_callers):
            depth = random.randint(300, 900)
            af = random.uniform(0.35, 0.65)
            raw_rows.append(
                {
                    "variant": vs,
                    "FILTER": random.choice(FILTER_CHOICES),
                    "TYPE": "SNP",
                    "FORMAT": "GT:AD:AF:DP:F1R2:F2R1:FAD:SB",
                    "GT_INFO": make_gt_info(af, depth),
                    "caller": caller,
                    "sample": sample,
                }
            )

# each sample's own unique low-VAF/mosaic variant, called by deepsomatic only
for sample, v in zip(samples, unique_variants):
    vs = variant_str(v)
    depth = random.randint(300, 900)
    af = random.uniform(0.01, 0.08)
    raw_rows.append(
        {
            "variant": vs,
            "FILTER": "PASS",
            "TYPE": "SNP",
            "FORMAT": "GT:AD:AF:DP:F1R2:F2R1:FAD:SB",
            "GT_INFO": make_gt_info(af, depth),
            "caller": "deepsomatic",
            "sample": sample,
        }
    )

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(f"{TEST_DATA}/raw.sample_call_key.tsv", sep="\t", index=False)

# ---------- sample_call_key.tsv (normalized key: og_vcf_key == variant) ----------
all_variants = sorted({variant_str(v) for v in background_variants + unique_variants})
norm_df = pd.DataFrame(
    {
        "variant": all_variants,
        "FILTER": ".",
        "TYPE": "SNP",
        "FORMAT": "GT",
        "GT_INFO": "0/1",
        "caller": "merged.decomposed",
        "og_vcf_key": all_variants,
    }
)
norm_df.to_csv(f"{TEST_DATA}/sample_call_key.tsv", sep="\t", index=False)

# ---------- per-sample readcount TSVs ----------
READCOUNT_COLS = [
    "sample", "chrom", "pos", "ref", "depth", "base", "count",
    "avg_mapping_quality", "avg_basequality", "avg_se_mapping_quality",
    "num_plus_strand", "num_minus_strand", "avg_pos_as_fraction",
    "avg_num_mismatches_as_fraction", "avg_sum_mismatch_qualities",
    "num_q2_containing_reads", "avg_distance_to_q2_start_in_q2_reads",
    "avg_clipped_length", "avg_distance_to_effective_3p_end",
]

# figure out which variants each sample has a call for, background + its own unique
sample_variants = {s: [] for s in samples}
for row in raw_rows:
    sample_variants[row["sample"]].append(row["variant"])
for s in sample_variants:
    sample_variants[s] = sorted(set(sample_variants[s]))

unique_variant_of_sample = {s: variant_str(v) for s, v in zip(samples, unique_variants)}

variant_lookup = {variant_str(v): v for v in background_variants + unique_variants}


def readcount_rows_for(sample, vs):
    chrom, pos, ref, alt = variant_lookup[vs]
    depth = random.randint(300, 900)
    is_mosaic = unique_variant_of_sample[sample] == vs
    af = random.uniform(0.01, 0.08) if is_mosaic else random.uniform(0.35, 0.65)
    alt_count = max(1, round(depth * af))
    ref_count = depth - alt_count

    def row(base, count):
        plus = round(count * random.uniform(0.45, 0.55))
        minus = count - plus
        return {
            "sample": sample,
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "depth": depth,
            "base": base,
            "count": count,
            "avg_mapping_quality": 60.0,
            "avg_basequality": round(random.uniform(29.5, 30.0), 2),
            "avg_se_mapping_quality": round(random.uniform(0.0, 0.3), 2),
            "num_plus_strand": plus,
            "num_minus_strand": minus,
            "avg_pos_as_fraction": round(random.uniform(0.35, 0.5), 2),
            "avg_num_mismatches_as_fraction": round(random.uniform(0.0, 0.02), 2),
            "avg_sum_mismatch_qualities": round(random.uniform(30, 50), 2),
            "num_q2_containing_reads": count,
            "avg_distance_to_q2_start_in_q2_reads": round(random.uniform(0.6, 0.7), 2),
            "avg_clipped_length": round(random.uniform(140, 150), 2),
            "avg_distance_to_effective_3p_end": round(random.uniform(0.4, 0.5), 2),
        }

    return [row(ref, ref_count), row(alt, alt_count)]


READCOUNT_DTYPES = {
    "sample": "category",
    "chrom": "category",
    "pos": "int32",
    "ref": "category",
    "depth": "int32",
    "base": "category",
    "count": "int32",
    "avg_mapping_quality": "float32",
    "avg_basequality": "float32",
    "avg_se_mapping_quality": "float32",
    "num_plus_strand": "int32",
    "num_minus_strand": "int32",
    "avg_pos_as_fraction": "float32",
    "avg_num_mismatches_as_fraction": "float32",
    "avg_sum_mismatch_qualities": "float32",
    "num_q2_containing_reads": "int32",
    "avg_distance_to_q2_start_in_q2_reads": "float32",
    "avg_clipped_length": "float32",
    "avg_distance_to_effective_3p_end": "float32",
}

os.makedirs(f"{TEST_DATA}/readcount", exist_ok=True)
for f in os.listdir(f"{TEST_DATA}/readcount"):
    os.remove(f"{TEST_DATA}/readcount/{f}")

for sample in samples:
    rows = []
    for vs in sample_variants[sample]:
        rows.extend(readcount_rows_for(sample, vs))
    df = pd.DataFrame(rows, columns=READCOUNT_COLS).astype(READCOUNT_DTYPES)
    df.to_parquet(
        f"{TEST_DATA}/readcount/{sample}_bam_readcount.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

# ---------- patient_info.xlsx / patient_info.tsv (generic values) ----------
# Both are generated from the same rows to demonstrate/exercise the two
# patient info formats app.py accepts (see load_patient_info in app.py).
# patient_info_path in config.toml picks which one gets loaded.
rows = []
for i, sample in enumerate(samples, start=1):
    rows.append(
        {
            "Patient ID": f"P{i:03d}",
            "Sample Status": "Affected",
            "Sex": "M" if i % 2 == 0 else "F",
            "Sequencing Name": sample,
            "Sequencing Round": 1,
        }
    )
pt_df = pd.DataFrame(rows)
pt_df.to_excel(f"{TEST_DATA}/patient_info.xlsx", sheet_name="Patient Information", index=False)
pt_df.to_csv(f"{TEST_DATA}/patient_info.tsv", sep="\t", index=False)

print("raw rows:", len(raw_df))
print("normalized rows:", len(norm_df))
print("samples:", len(samples))
print("done")
