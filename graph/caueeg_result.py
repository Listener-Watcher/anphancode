import os
import pandas as pd

# folders = [
#     "result_Mar25_MIL-LinkX",
#     "result_Mar31_MIL-LinkX",
#     "result_Apr01",
#     "result_Apr02_Baseline/master_full_data_bi23_250hz.h5",
#     "result_Apr02_Baseline/master_full_data_mono_250hz.h5",
# ]

# base_dir = "/home/anphan/Documents/EEG_Project/AHEAP_data"
# output_excel = os.path.join(base_dir, "all_results_summary.xlsx")

# aggregated_data = []

# for name_folder in folders:
#     result_folder = os.path.join(base_dir, name_folder)

#     if not os.path.exists(result_folder):
#         print(f"Skip, folder not found: {result_folder}")
#         continue
#     for root, dirs, files in os.walk(result_folder):
#         if "overall_summary_test.csv" not in files:
#             continue

#         csv_path = os.path.join(root, "overall_summary_test.csv")

#         try:
#             df = pd.read_csv(csv_path, index_col=0)

#             # Example expected index: mean, std
#             # Example expected columns: accuracy, balanced_accuracy, macro_f1
#             row = {
#                 "id_folder": os.path.basename(root),
#                 "parent_folder": name_folder,
#                 "avg_acc": df.loc["mean", "accuracy"],
#                 "avg_bal_acc": df.loc["mean", "balanced_accuracy"],
#                 "avg_f1": df.loc["mean", "macro_f1"],
#                 "std_acc": df.loc["std", "accuracy"],
#                 "std_bal_acc": df.loc["std", "balanced_accuracy"],
#                 "std_f1": df.loc["std", "macro_f1"],
#                 "csv_path": csv_path,
#             }

#             aggregated_data.append(row)
#             print(f"Added: {root}")

#         except Exception as e:
#             print(f"Error reading {csv_path}: {e}")

# df_aggregated = pd.DataFrame(aggregated_data)

# # Optional: sort rows
# df_aggregated = df_aggregated.sort_values(["parent_folder", "id_folder"]).reset_index(drop=True)

# # Save to Excel
# df_aggregated.to_excel(output_excel, index=False)

# print(f"Aggregated results saved to {output_excel}")
# print(df_aggregated)

import os
import pandas as pd

folders = [
    # "result_MIL-LinkX",
    # "result_caueeg_linkx_segment",

]

base_dir = "/home/anphan/Documents/EEG_Project/CAUEEG/results_caueeg_linkx_segment"
# base_dir = "/home/anphan/Documents/EEG_Project/CAUEEG/result_MIL-LinkX-update-correctchannel"
output_excel = os.path.join(base_dir, "caueeg_segment_summary.xlsx")
# target_csv_name = "summary_test.csv"
target_csv_name = "summary_metrics.csv"


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
if os.path.exists(output_excel):
    df_master = pd.read_excel(output_excel)
else:
    df_master = pd.DataFrame()

# -------------------------------------------------
# 2. Backward compatibility: create result_key if missing
# -------------------------------------------------
if not df_master.empty and "result_key" not in df_master.columns:
    print("Old summary file detected. Creating result_key column...")
    df_master["result_key"] = df_master.apply(
        lambda row: build_result_key_from_row(row, base_dir),
        axis=1
    )

# Make sure required columns exist even for old files
required_cols = [
    "result_key",
    "parent_folder",
    "id_folder",
    "avg_acc",
    "avg_bal_acc",
    "avg_f1",
]

for col in required_cols:
    if col not in df_master.columns:
        df_master[col] = pd.NA

# Optional: remove duplicates already present in old file
df_master = df_master.drop_duplicates(subset="result_key", keep="last").reset_index(drop=True)

# -------------------------------------------------
# 3. Scan current folders
# -------------------------------------------------
new_rows = []

# for name_folder in folders:
#     result_folder = os.path.join(base_dir, name_folder)

#     if not os.path.exists(result_folder):
#         print(f"Skip, folder not found: {result_folder}")
#         continue

#     for root, dirs, files in os.walk(result_folder):
#         if "overall_summary_test.csv" not in files:
#             continue

for root, dirs, files in os.walk(base_dir):
    if target_csv_name not in files:
        continue

    csv_path = os.path.join(root, target_csv_name)

        # csv_path = os.path.join(root, "overall_summary_test.csv")

    try:
        df = pd.read_csv(csv_path, index_col=0)

        result_key = os.path.relpath(root, base_dir)

        # first folder under base_dir
        rel_parts = os.path.relpath(root, base_dir).split(os.sep)
        parent_folder = rel_parts[0] if len(rel_parts) > 0 else ""
        id_folder = os.path.basename(root)

        row = {
            "result_key": result_key,
            "parent_folder": parent_folder,
            "id_folder": id_folder,
            "avg_acc": df.loc["test", "accuracy"],
            "avg_bal_acc": df.loc["test", "balanced_accuracy"],
            "avg_f1": df.loc["test", "macro_f1"],
            # "csv_path": csv_path,
            # "csv_mtime": os.path.getmtime(csv_path),
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
df_master = df_master.sort_values(["parent_folder", "id_folder"], na_position="last").reset_index(drop=True)
df_master.to_excel(output_excel, index=False)

print(f"Updated summary saved to: {output_excel}")
print(df_master)