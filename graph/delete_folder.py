import os
import csv
import shutil

csv_file = "/home/anphan/Documents/EEG_Project/delete_folders.csv"
dry_run = False   # change to False when ready to actually delete
root = "/home/anphan/Documents/EEG_Project/CAUEEG"
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        parent_folder = row["parent_folder"].strip()
        folder_name = row["folder_name"].strip()

        folder_path = os.path.join(root, parent_folder, folder_name)
        folder_path = os.path.abspath(folder_path)

        # extra safety checks
        if folder_path in ["/", "/home", "/home/anphan"]:
            print(f"SKIP unsafe path: {folder_path}")
            continue

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            if dry_run:
                print(f"[DRY RUN] Would delete: {folder_path}")
            else:
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
        else:
            print(f"Not found or not a folder: {folder_path}")