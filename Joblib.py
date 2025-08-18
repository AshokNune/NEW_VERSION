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
        return df
    except Exception as e:
        print(f"❌ Skipping {file}: {e}")
        return None

def folder_excel_to_parquet_joblib(input_folder, output_parquet, columns_to_keep, n_jobs=-1):
    """
    Reads all Excel files in a folder in parallel (with joblib), keeps only the specified columns,
    concatenates them, and writes to a Parquet file.
    Shows progress bar with tqdm.
    """
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx"))
    if not all_files:
        print("⚠️ No Excel files found in the folder.")
        return
    
    # tqdm for progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_excel)(file, columns_to_keep) 
        for file in tqdm(all_files, desc="Processing Excel files", unit="file")
    )
    
    # Filter out None results
    df_list = [df for df in results if df is not None]
    
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_parquet(output_parquet, index=False)
        print(f"✅ Saved combined parquet: {output_parquet}")
    else:
        print("⚠️ No valid data to save.")

# Example usage
if __name__ == "__main__":
    input_folder = "input_excels"   # folder with .xlsx files
    output_file = "final_output.parquet"
    required_columns = ["Text", "Keyword", "Classification"]  # change as needed
    
    folder_excel_to_parquet_joblib(input_folder, output_file, required_columns, n_jobs=-1)
