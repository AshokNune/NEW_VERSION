import pandas as pd
import os
import glob
from joblib import Parallel, delayed
from tqdm import tqdm

def process_excel(file, columns_to_keep):
    """Read an Excel file and return only required columns."""
    try:
        df = pd.read_excel(file, engine="openpyxl")
        df = df[columns_to_keep]
        print(f"📄 Processed: {os.path.basename(file)}")
        return df
    except Exception as e:
        print(f"❌ Skipping {file}: {e}")
        return None

def consolidate_excels(input_folder, output_file, required_columns, n_jobs=-1):
    """Consolidate all Excel files in a folder into a parquet file."""
    # Load existing parquet if it exists
    if os.path.exists(output_file):
        existing_df = pd.read_parquet(output_file)
        print(f"📂 Existing parquet loaded with {len(existing_df)} rows.")
    else:
        existing_df = pd.DataFrame(columns=required_columns)
        print("📂 No existing parquet found. Starting fresh.")

    # Process new Excel files
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    if not all_files:
        print("⚠️ No Excel files found in the folder.")
        return

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_excel)(file, required_columns) 
        for file in tqdm(all_files, desc="Processing Excel files", unit="file")
    )
    df_list = [df for df in results if df is not None]

    if df_list:
        new_df = pd.concat(df_list, ignore_index=True)
        print(f"🆕 New data rows: {len(new_df)}")

        # Combine with existing parquet
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"📊 Total rows before grouping: {len(combined_df)}")

        # Group by "path" and sum "coun"
        combined_df = combined_df.groupby("path", as_index=False)["coun"].sum()

        print(f"📊 Total rows after grouping: {len(combined_df)}")
        combined_df.to_parquet(output_file, index=False)
        print(f"✅ Parquet updated: {output_file}")
    else:
        print("⚠️ No valid new data to append.")

# ================= MAIN BLOCK =================
if __name__ == "__main__":
    input_folder = "input_excels"   # folder with .xlsx files
    output_file = "final_output.parquet"
    required_columns = ["path", "coun"]  # columns to keep
    n_jobs = -1  # use all CPU cores

    consolidate_excels(input_folder, output_file, required_columns, n_jobs)
