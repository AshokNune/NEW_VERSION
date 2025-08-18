import pandas as pd
import os
import glob
import hashlib
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime

# ------------------- Helper Functions -------------------

def file_hash(file_path):
    """Return MD5 hash of a file for quick change detection."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_row_hash(df, columns_to_keep):
    """Add a row-level hash for deduplication using fast pandas hashing."""
    df["__row_hash__"] = pd.util.hash_pandas_object(df[columns_to_keep], index=False).astype(str)
    return df

def process_excel(file, columns_to_keep):
    """Read an Excel file, keep required columns, and add row hashes."""
    try:
        df = pd.read_excel(file, engine="openpyxl")
        df = df[columns_to_keep]
        df = add_row_hash(df, columns_to_keep)
        df["__source_file__"] = os.path.basename(file)
        df["__file_hash__"] = file_hash(file)
        return df
    except Exception as e:
        print(f"❌ Skipping {file}: {e}")
        return None

def generate_summary(parquet_file, summary_file, group_col="__source_file__", unique_col="unique_id"):
    """
    Generate summary CSV with count of unique values grouped by a specified column.
    Overwrites summary_file on each run.
    """
    if not os.path.exists(parquet_file):
        print("⚠️ No parquet file found for summary.")
        return

    df = pd.read_parquet(parquet_file)

    if group_col not in df.columns or unique_col not in df.columns:
        print(f"⚠️ Columns '{group_col}' or '{unique_col}' not found in parquet. Cannot generate summary.")
        return

    summary = (
        df.groupby(group_col)[unique_col]
        .nunique()
        .reset_index()
        .rename(columns={unique_col: "unique_count"})
    )

    summary.to_csv(summary_file, index=False)
    print(f"📊 Summary saved (overwrite): {summary_file}")

# ------------------- Main Function -------------------

def folder_excel_to_parquet_full_pipeline(input_folder, output_parquet, log_file,
                                          columns_to_keep, group_col="__source_file__",
                                          unique_col="unique_id", n_jobs=-1):
    """
    Full pipeline:
    - Incremental parquet update (file + row level dedup)
    - Logging of processed files
    - Summary CSV generation
    """
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    if not all_files:
        print("⚠️ No Excel files found in the folder.")
        return

    # Load existing parquet
    existing_df = None
    file_hash_map = {}
    if os.path.exists(output_parquet):
        try:
            existing_df = pd.read_parquet(output_parquet)
            file_hash_map = dict(zip(existing_df["__source_file__"], existing_df["__file_hash__"]))
            print(f"📂 Found existing parquet with {len(existing_df)} rows, {len(file_hash_map)} files tracked.")
        except Exception as e:
            print(f"⚠️ Could not read existing parquet: {e}")

    logs = []

    # Detect new, modified, unchanged files
    new_or_modified_files = []
    for f in all_files:
        fname = os.path.basename(f)
        fhash = file_hash(f)

        if fname not in file_hash_map:
            logs.append([datetime.now(), fname, fhash, "new", 0])
            new_or_modified_files.append(f)
        elif file_hash_map[fname] != fhash:
            logs.append([datetime.now(), fname, fhash, "modified", 0])
            new_or_modified_files.append(f)
        else:
            logs.append([datetime.now(), fname, fhash, "unchanged", 0])

    if not new_or_modified_files:
        print("✅ No new or modified files to process. Parquet is already up to date.")
    else:
        print(f"🔍 Found {len(new_or_modified_files)} new/modified files to process.")

        # Process files in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_excel)(file, columns_to_keep)
            for file in tqdm(new_or_modified_files, desc="Processing Excel files", unit="file")
        )

        # Collect valid DataFrames
        new_dfs = [df for df in results if df is not None]
        if new_dfs:
            new_data = pd.concat(new_dfs, ignore_index=True)

            # Update row counts in logs
            for fname in new_data["__source_file__"].unique():
                row_count = len(new_data[new_data["__source_file__"] == fname])
                for log in logs:
                    if log[1] == fname:
                        log[4] = row_count

            # Merge with existing parquet
            if existing_df is not None:
                modified_files = set(new_data["__source_file__"].unique())
                updated_existing = existing_df[~existing_df["__source_file__"].isin(modified_files)]
                final_df = pd.concat([updated_existing, new_data], ignore_index=True)
                final_df = final_df.drop_duplicates(subset=["__source_file__", "__row_hash__"], keep="last")
            else:
                final_df = new_data

            final_df.to_parquet(output_parquet, index=False)
            print(f"✅ Updated parquet saved with {len(final_df)} rows: {output_parquet}")

    # Save log (append if exists)
    log_df = pd.DataFrame(logs, columns=["timestamp", "file", "file_hash", "action", "rows"])
    if os.path.exists(log_file):
        old_logs = pd.read_csv(log_file)
        log_df = pd.concat([old_logs, log_df], ignore_index=True)
    log_df.to_csv(log_file, index=False)
    print(f"📝 Log updated: {log_file}")

    # Generate summary CSV (overwrite)
    summary_file = "summary.csv"
    generate_summary(output_parquet, summary_file, group_col=group_col, unique_col=unique_col)

# ------------------- Example Usage -------------------

if __name__ == "__main__":
    input_folder = "input_excels"
    output_file = "final_output.parquet"
    log_file = "processing_log.csv"
    required_columns = ["Text", "Keyword", "Classification", "unique_id"]  # ensure unique_id exists

    folder_excel_to_parquet_full_pipeline(
        input_folder=input_folder,
        output_parquet=output_file,
        log_file=log_file,
        columns_to_keep=required_columns,
        group_col="__source_file__",
        unique_col="unique_id",
        n_jobs=-1
    )
