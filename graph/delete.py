# import shutil
# import os

# folders = [
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_plv_alpha_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_corr_alpha_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_coherence_alpha_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_pli_alpha_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_plv_None_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_corr_None_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_coherence_None_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/mono_rbphjorth_pli_None_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/bi23_rbp_plv_None_25edges',
#   '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/bi23_rbp_plv_None_35edges'
# ]

# for folder in folders:
#   if os.path.exists(folder):
#       shutil.rmtree(folder)
#       print(f"Deleted: {folder}")
#   else:
#       print(f"Not found: {folder}")
# import os

# # root_dir = "/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data"   # change this
# root_dir = "/home/anphan/Documents/EEG_Project/AHEAP_data/result_Mar31_MIL-LinkX"   # change this


# for d in os.listdir(root_dir):
#   path = os.path.join(root_dir, d)
#   if os.path.isdir(path):
#       # print(path)
#       last_part = os.path.basename(path)
#       parts = last_part.split('_')
#       # if len(parts) > 4:
#       print(f"{last_part}")
# # for root, dirs, files in os.walk(root_dir):
# #     for d in dirs:
# #         full_path = os.path.join(root, d)
# #         print(full_path)
import os
import fnmatch

base_dir = "/home/anphan/Documents/CAUEEG/result_new-arc"   # change if needed
# base_dir = "/home/anphan/Documents/EEG_Project/AHEAP_data"   # change if needed
folders = []
for d in os.listdir(base_dir):
    path = os.path.join(base_dir, d)
    if os.path.isdir(path):
        # print(path)
        last_part = os.path.basename(path)
        folders.append(last_part)
#       parts = last_part.split('_')
#       # if len(parts) > 4:
#       print(f"{last_part}")
# folders = [
#   "result_Mar25_MIL-LinkX",
#   "result_Mar31_MIL-LinkX",
#   "result_Apr01",
#   "result_Apr01_bi23",
#   "result_Apr03",
#   "result_Apr06",
#   "result_Apr06-mlpnode",
#   "result_Apr07-mlpedge",
#   "result_Apr07-LINKXhypergraph",
#   "result_Apr08_nodetopology_zscoredata",
#   "result_Apr08_zscoredata_residualconn",
#   "result_Apr09_nodetopology_zscoredata",
#   "result_Apr09_zscoredata",
#   "result_Apr05_zscoredata",
#   "result_Apr02_Baseline/master_full_data_bi23_250hz.h5",
#   "result_Apr02_Baseline/master_full_data_mono_250hz.h5",
# ]

dry_run = False   # set False after checking

total_found = 0
total_deleted = 0
csv_name = "aggregate_seed_results.csv"
# csv_name = "summary_metrics.csv"
# csv_name = "overall_summary_test.csv"
for name_folder in folders:
    result_folder = os.path.join(base_dir, name_folder)
    # print(result_folder)

    if not os.path.exists(result_folder):
        print(f"Skip, folder not found: {result_folder}")
        continue


    # only process finished result folders
    # summary_csv = os.path.join(result_folder, "agg_seed_results.csv", csv_name)
    # if not os.path.isfile(summary_csv):
    #     print(f"Skip, missing {csv_name}: {result_folder}")
    #     continue

    print(f"\nScanning: {result_folder}")

    for item in os.listdir(result_folder):
        print(item)
        if ".csv" in item:
            continue
        if ".png" in item:
            continue
        # if ".pt" in item:
        #     continue
        if ".txt" in item:
            continue
        if ".json" in item:
            continue
        item_path = os.path.join(result_folder, item)
        print(item_path)

    

        topk_files = [
            f for f in os.listdir(result_folder)
            if fnmatch.fnmatch(f, "*_topk_epoch*.pt")
        ]

        if not topk_files:
            print(f"  No top-k files in: {item_path}")
            continue

        print(f"  Found {len(topk_files)} top-k file(s) in: {item_path}")
        total_found += len(topk_files)

        for fname in topk_files:
            fpath = os.path.join(result_folder, fname)
            # fpath = os.path.join(item_path, fname)

            if dry_run:
                print(f"    [DRY RUN] Would delete: {fpath}")
            else:
                try:
                    os.remove(fpath)
                    total_deleted += 1
                    print(f"    Deleted: {fpath}")
                except Exception as e:
                    print(f"    Error deleting {fpath}: {e}")

        # for seed_path in os.listdir(item_path):
        #   print(seed_path)
        #   if not seed_path.startswith("seed"):
        #       continue

        #   checkpoints_dir = os.path.join(item_path, seed_path, "checkpoints")
        #   if not os.path.isdir(checkpoints_dir):
        #       print(f"  Skip, no checkpoints folder: {checkpoints_dir}")
        #       continue

            # topk_file = [
            #   f for f in os.listdir(checkpoints_dir)
            #   if fnmatch.fnmatch(f, "*_topk_epoch*.pt")
            # ]
         
            # # checkpoints_dir = os.path.join(item_path, seed_path)
            # # image_files = [
            # #     f for f in os.listdir(os.path.join(item_path, seed_path))
            # #     if fnmatch.fnmatch(f, "tsne_fold*_segment_embeddings_tsne.png")
            # # ]
            # topk_files = topk_file #+ image_files

            # if not topk_files:
            #   print(f"  No top-k files in: {checkpoints_dir}")
            #   continue

            # print(f"  Found {len(topk_files)} top-k file(s) in: {checkpoints_dir}")
            # total_found += len(topk_files)

            # for fname in topk_files:
            #   fpath = os.path.join(checkpoints_dir, fname)

            #   if dry_run:
            #       print(f"    [DRY RUN] Would delete: {fpath}")
            #   else:
            #       try:
            #           os.remove(fpath)
            #           total_deleted += 1
            #           print(f"    Deleted: {fpath}")
            #       except Exception as e:
            #           print(f"    Error deleting {fpath}: {e}")

print("\nDone.")
print(f"Total matched top-k files: {total_found}")
if dry_run:
    print("Dry run only. No files were deleted.")
else:
    print(f"Total deleted: {total_deleted}")
