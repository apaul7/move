import argparse
import pandas as pd
import pyarrow as pa

DTYPES = {
    'sample': 'category',
    'chrom': 'category',
    'pos': 'int32',
    'ref': 'category',
    'depth': 'int32',
    'base': 'category',
    'count': 'int32',
    'avg_mapping_quality': 'float32',
    'avg_basequality': 'float32',
    'avg_se_mapping_quality': 'float32',
    'num_plus_strand': 'int32',
    'num_minus_strand': 'int32',
    'avg_pos_as_fraction': 'float32',
    'avg_num_mismatches_as_fraction': 'float32',
    'avg_sum_mismatch_qualities': 'float32',
    'num_q2_containing_reads': 'int32',
    'avg_distance_to_q2_start_in_q2_reads': 'float32',
    'avg_clipped_length': 'float32',
    'avg_distance_to_effective_3p_end': 'float32',
}

# Mirrors DTYPES. The category columns are dictionary-encoded rather than plain
# strings so they land in pandas as Categoricals directly -- as strings they'd
# be one Python str object per cell, which costs more than every numeric column
# combined. Chunks build their own dictionaries; concat_tables unifies them.
_ARROW_TYPES = {
    'int32': pa.int32(),
    'float32': pa.float32(),
    'category': pa.dictionary(pa.int32(), pa.string()),
}
ARROW_SCHEMA = pa.schema([(name, _ARROW_TYPES[dtype]) for name, dtype in DTYPES.items()])
CATEGORY_COLUMNS = [name for name, dtype in DTYPES.items() if dtype == 'category']

# Rows are converted to Arrow this many at a time, so a whole readcount file is
# never resident as Python objects (~1kB/row) -- only as Arrow buffers (~76B/row).
CHUNK_ROWS = 250_000


def parse_base(sample, chrom, pos, ref, depth, values):
    """One bam-readcount base field -> a tuple in DTYPES column order."""
    return (
        sample,
        chrom,
        pos,
        ref,
        depth,
        values[0],
        int(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        int(values[5]),
        int(values[6]),
        float(values[7]),
        float(values[8]),
        float(values[9]),
        int(values[10]),
        float(values[11]),
        float(values[12]),
        float(values[13]),
    )


def iter_rows(file_path, sample):
    with open(file_path) as file:
        for line in file:
            row = line.rstrip().split('\t')
            if len(row) < 9:
                raise ValueError(
                    f"{file_path}: expected at least 9 tab-separated columns, "
                    f"got {len(row)}: {line[:80]!r}"
                )
            chrom, pos, ref, depth = row[0], int(row[1]), row[2], int(row[3])

            # row[4] is the '=' pseudo-base and is dropped; row[5:9] are A/C/G/T.
            for field in row[5:9]:
                values = field.split(':')
                if int(values[1]) == 0:
                    continue
                yield parse_base(sample, chrom, pos, ref, depth, values)

            # everything past row[9] is indels (and N, which is dropped)
            for field in row[9:]:
                if field[0] == 'N':
                    continue
                yield parse_base(sample, chrom, pos, ref, depth, field.split(':'))


def to_arrow(batch):
    columns = zip(*batch)  # rows -> columns
    return pa.Table.from_arrays(
        [pa.array(col, type=field.type) for col, field in zip(columns, ARROW_SCHEMA)],
        schema=ARROW_SCHEMA,
    )


def read_chunks(file_path, sample):
    """Arrow tables of at most CHUNK_ROWS rows each."""
    batch = []
    for row in iter_rows(file_path, sample):
        batch.append(row)
        if len(batch) >= CHUNK_ROWS:
            yield to_arrow(batch)
            batch = []
    if batch:
        yield to_arrow(batch)


def main():
    parser = argparse.ArgumentParser(description='Merge snv and indel bam-readcount output to a single parquet file')
    parser.add_argument('--sample',  type=str, help='Sample name')
    parser.add_argument('--snv', type=str, help='Path to snv bam-readcount file')
    parser.add_argument('--indel', type=str, help='Path to indel bam-readcount file')
    parser.add_argument('--out', type=str, help='Output filename (parquet)')

    args = parser.parse_args()

    # the chunk list is a temporary, so concat_tables' inputs are released before
    # to_pandas runs. self_destruct then frees each Arrow buffer as it converts,
    # rather than holding the full table and the full DataFrame at once -- which
    # needs split_blocks, otherwise pandas consolidates into one block up front
    # and every buffer stays referenced until the whole conversion is done.
    table = pa.concat_tables(
        [
            *read_chunks(args.snv, args.sample),
            *read_chunks(args.indel, args.sample),
        ]
    )
    readcount_df = table.to_pandas(self_destruct=True, split_blocks=True)
    del table

    # ponytail: the frame is still built whole in memory -- the dedup below is
    # global, so nothing can be written until every line is parsed. If a single
    # sample ever outgrows RAM, sort by (chrom, pos) and stream out one position
    # at a time with pyarrow.parquet.ParquetWriter.
    readcount_df = readcount_df.astype(DTYPES)

    # Arrow orders dictionary values by first appearance; pandas' own
    # astype('category') sorts them. Match pandas, so the categories written to
    # the parquet are the same as before this script streamed through Arrow.
    for column in CATEGORY_COLUMNS:
        values = readcount_df[column]
        readcount_df[column] = values.cat.reorder_categories(sorted(values.cat.categories))

    readcount_df = readcount_df.drop_duplicates()

    readcount_df.to_parquet(args.out, engine='pyarrow', compression='snappy', index=False)

if __name__ == "__main__":
    main()
