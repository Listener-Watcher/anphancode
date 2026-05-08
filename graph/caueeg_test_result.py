import os
import pandas as pd
from pathlib import Path

base_dir = "/home/anphan/Documents/CAUEEG"
path = Path(base_dir)

subfolders = list(path.glob("result*"))
# base_dir = "/home/anphan/Documents/CAUEEG/result_MIL-LinkX-weighted"
# base_dir = "/home/anphan/Documents/CAUEEG/result_dual_branch_clean_grid"
# base_dir = "/home/anphan/Documents/EEG_Project/CAUEEG/result_MIL-LinkX-update-correctchannel"
# output_excel = os.path.join(base_dir, "caueeg_results_test_config.xlsx")
# target_csv_name = "summary_test.csv"


def build_result_key_from_row(row, base_dir):
    # Best case: recover from csv_path
    if "csv_path" in row and pd.notna(row["csv_path"]):
        csv_path = str(row["csv_path"])
        root = os.path.dirname(csv_path)
        return os.path.relpath(root, base_dir)

    # Next best: parent_folder + id_folder
    if "parent_folder" in row and "id_folder" in row:
        parent_folder = str(row["parent_folder"])
        id_folder = str(row["id_folder"])
        return os.path.join(parent_folder, id_folder)

    # Last fallback: id_folder only
    if "id_folder" in row:
        return str(row["id_folder"])

    raise ValueError("Cannot build result_key from old summary file.")


# -------------------------------------------------
# 1. Load old summary file
# -------------------------------------------------
df_all = pd.DataFrame()

for sub_dir in subfolders:
    print(sub_dir)
    output_excel = os.path.join(sub_dir, "caueeg_results_paper.xlsx")
    target_csv_name = "aggregate_seed_results.csv"
    df_master = pd.DataFrame()

    # -------------------------------------------------
    # 2. Backward compatibility: create result_key if missing
    # -------------------------------------------------
    if not df_master.empty and "result_key" not in df_master.columns:
        print("Old summary file detected. Creating result_key column...")
        df_master["result_key"] = df_master.apply(
            lambda row: build_result_key_from_row(row, sub_dir),
            axis=1
        )

    # Make sure required columns exist even for old files
    required_cols = [
        "result_key",
        "parent_folder",
        "id_folder",
    ]

    for col in required_cols:
        if col not in df_master.columns:
            df_master[col] = pd.NA
    # Optional: remove duplicates already present in old file
    df_master = df_master.drop_duplicates(subset="result_key", keep="last").reset_index(drop=True)
    new_rows = []

    for root, dirs, files in os.walk(sub_dir):
        if target_csv_name not in files:
            continue

        csv_path = os.path.join(root, target_csv_name)

        try:
            df = pd.read_csv(csv_path, index_col=0)

            result_key = os.path.relpath(root, sub_dir)
            rel_parts = os.path.relpath(root, sub_dir).split(os.sep)
            parent_folder = rel_parts[0] if len(rel_parts) > 0 else ""
            id_folder = os.path.basename(root)
            if id_folder.startswith("agg_seed_results"):
                id_folder = os.path.basename(os.path.dirname(root))
            # id_folder = os.path.basename(root)

            row = {
                "result_key": result_key,
                "parent_folder": parent_folder,
                "id_folder": id_folder,
                # **df.iloc[0].to_dict(),
                **df.reset_index().iloc[0].to_dict(),
            }

            new_rows.append(row)
            print(f"Found: {csv_path}")

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    df_new = pd.DataFrame(new_rows)

    # -------------------------------------------------
    # 4. Upsert
    # -------------------------------------------------
    if not df_new.empty:
        df_master = df_master[~df_master["result_key"].isin(df_new["result_key"])]
        df_master = pd.concat([df_master, df_new], ignore_index=True)

    # -------------------------------------------------
    # 5. Save
    # -------------------------------------------------
    df_master = df_master.drop(["result_key"], axis=1)
    df_master = df_master.sort_values(["parent_folder", "id_folder"], na_position="last").reset_index(drop=True)
    df_master.to_excel(output_excel, index=False)

    print(f"Updated summary saved to: {output_excel}")
    print(df_master)
    df_all = pd.concat([df_all, df_master], ignore_index=True)
df_all.to_excel("/home/anphan/Documents/CAUEEG/all_paper.xlsx", index=False)
