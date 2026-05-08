


# merge_h5_random_train_sliding_eval.py

import argparse
import json
from pathlib import Path
from collections import Counter

import h5py


def copy_root_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def copy_non_subject_root_items(src, dst, skip_existing=True):
    """
    Copy root-level groups/datasets except /subjects.
    Usually this preserves global metadata if your master H5 has any.
    """
    for key in src.keys():
        if key == "subjects":
            continue

        if key in dst:
            if skip_existing:
                continue
            del dst[key]

        src.copy(key, dst, name=key)


def select_subject_ids(h5f, prefixes):
    if "subjects" not in h5f:
        raise KeyError(f"H5 file does not contain root group 'subjects'. Keys={list(h5f.keys())}")

    prefixes = tuple(prefixes)
    return sorted([sid for sid in h5f["subjects"].keys() if sid.startswith(prefixes)])


def copy_subjects(src, dst, subject_ids, source_name):
    copied = []
    skipped = []

    for sid in subject_ids:
        if sid in dst["subjects"]:
            skipped.append(sid)
            continue

        src.copy(f"subjects/{sid}", dst["subjects"], name=sid)
        copied.append(sid)

    return {
        "source": source_name,
        "requested": len(subject_ids),
        "copied": len(copied),
        "skipped_existing": len(skipped),
        "copied_subject_ids": copied,
        "skipped_subject_ids": skipped,
    }


def count_splits(subject_ids):
    counts = Counter()
    for sid in subject_ids:
        if sid.startswith("train_"):
            counts["train"] += 1
        elif sid.startswith("val_"):
            counts["val"] += 1
        elif sid.startswith("validation_"):
            counts["validation"] += 1
        elif sid.startswith("test_"):
            counts["test"] += 1
        else:
            counts["unknown"] += 1
    return dict(counts)


def merge_h5(
    random_crop_h5,
    sliding_window_h5,
    output_h5,
    overwrite=False,
    val_prefixes=("val_", "validation_"),
    test_prefixes=("test_",),
):
    random_crop_h5 = Path(random_crop_h5)
    sliding_window_h5 = Path(sliding_window_h5)
    output_h5 = Path(output_h5)

    if output_h5.exists():
        if overwrite:
            output_h5.unlink()
        else:
            raise FileExistsError(
                f"Output file already exists: {output_h5}. "
                f"Use --overwrite if you want to replace it."
            )

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "random_crop_h5": str(random_crop_h5),
        "sliding_window_h5": str(sliding_window_h5),
        "output_h5": str(output_h5),
        "rule": {
            "train": "from random_crop_h5, subject_id starts with train_",
            "val": f"from sliding_window_h5, subject_id starts with {val_prefixes}",
            "test": f"from sliding_window_h5, subject_id starts with {test_prefixes}",
        },
        "copy_reports": [],
    }

    with h5py.File(random_crop_h5, "r") as fr, \
         h5py.File(sliding_window_h5, "r") as fs, \
         h5py.File(output_h5, "w") as fo:

        # Copy root attrs and any global metadata from random-crop file first.
        copy_root_attrs(fr, fo)
        copy_non_subject_root_items(fr, fo)

        # Create subject root group.
        fo.create_group("subjects")

        train_ids = select_subject_ids(fr, ("train_",))
        val_ids = select_subject_ids(fs, val_prefixes)
        test_ids = select_subject_ids(fs, test_prefixes)

        manifest["selected_counts"] = {
            "train_from_random_crop": len(train_ids),
            "val_from_sliding_window": len(val_ids),
            "test_from_sliding_window": len(test_ids),
        }

        manifest["copy_reports"].append(copy_subjects(fr, fo, train_ids, "random_crop_train"))
        manifest["copy_reports"].append(copy_subjects(fs, fo, val_ids, "sliding_window_val"))
        manifest["copy_reports"].append(copy_subjects(fs, fo, test_ids, "sliding_window_test"))

        final_ids = sorted(list(fo["subjects"].keys()))
        manifest["final_num_subjects"] = len(final_ids)
        manifest["final_split_counts"] = count_splits(final_ids)

        # Store merge manifest inside output H5.
        fo.attrs["merge_manifest_json"] = json.dumps(manifest, indent=2)

    manifest_path = output_h5.with_suffix(".merge_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved merged H5:", output_h5)
    print("Saved manifest:", manifest_path)
    print("Selected counts:", manifest["selected_counts"])
    print("Final split counts:", manifest["final_split_counts"])

    return manifest


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--random_crop_h5", required=True)
    # parser.add_argument("--sliding_window_h5", required=True)
    # parser.add_argument("--output_h5", required=True)
    # parser.add_argument("--overwrite", action="store_true")
    # args = parser.parse_args()
    random_crop_h5 = "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    sliding_window_h5 = "/home/anphan/Documents/caueeg_sliding_mono_dementia_seed42_overlap50.h5"
    output_h5 = "/home/anphan/Documents/caueeg_random_train_sliding_valtest_mono_dementia_seed42_overlap50.h5"
    overwrite = False
    merge_h5(
        random_crop_h5=random_crop_h5,
        sliding_window_h5=sliding_window_h5,
        output_h5=output_h5,
        overwrite=overwrite,
    )

####################################3 correct the mismatch label
# import json
# import shutil
# from pathlib import Path

# import h5py


# def patch_one_h5_label_file(
#     h5_path: str | Path,
#     *,
#     old_to_new: dict[int, int],
#     make_backup: bool = True,
#     backup_suffix: str = ".bak",
#     write_label_mapping_attr: bool = True,
# ) -> None:
#     h5_path = Path(h5_path)

#     if not h5_path.exists():
#         raise FileNotFoundError(f"Missing file: {h5_path}")

#     if make_backup:
#         backup_path = h5_path.with_name(h5_path.name + backup_suffix)
#         if not backup_path.exists():
#             shutil.copy2(h5_path, backup_path)

#     with h5py.File(h5_path, "r+") as h5f:
#         if "subjects" not in h5f:
#             raise KeyError(f"{h5_path} does not contain /subjects")

#         changed = 0
#         label_counter_before = {}
#         label_counter_after = {}

#         for sid in h5f["subjects"].keys():
#             meta = h5f["subjects"][sid]["metadata"]

#             old_label = int(meta.attrs["label"])
#             if old_label not in old_to_new:
#                 raise KeyError(
#                     f"{h5_path}: subject {sid} has label {old_label}, "
#                     f"but it is not in old_to_new={old_to_new}"
#                 )

#             new_label = int(old_to_new[old_label])

#             label_counter_before[old_label] = label_counter_before.get(old_label, 0) + 1
#             label_counter_after[new_label] = label_counter_after.get(new_label, 0) + 1

#             meta.attrs["label"] = new_label
#             meta.attrs["class_id"] = new_label
#             changed += 1

#         if write_label_mapping_attr:
#             h5f.attrs["label_to_int_json"] = json.dumps(
#                 {"C": 0, "A": 1, "F": 2},
#                 ensure_ascii=False,
#             )

#     print(f"Patched: {h5_path}")
#     print(f"  Subjects changed: {changed}")
#     print(f"  Before: {label_counter_before}")
#     print(f"  After : {label_counter_after}")


# def patch_all_master_h5_under_root(
#     root_dir: str | Path,
#     *,
#     pattern: str = "**/master.h5",
#     old_to_new: dict[int, int] | None = None,
#     make_backup: bool = True,
# ) -> None:
#     root_dir = Path(root_dir)

#     if old_to_new is None:
#         # current wrong encoding: A->0, C->1, F->2
#         # desired encoding:       A->1, C->0, F->2
#         old_to_new = {0: 1, 1: 0, 2: 2}

#     h5_files = sorted(root_dir.glob(pattern))
#     if not h5_files:
#         raise FileNotFoundError(f"No files matched {pattern!r} under {root_dir}")

#     # already_corrected = [
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration10.0_overlap0.5/master.h5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/mono_duration10.0_overlap0.5/master.h5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration10.0_overlap0.8/master.h5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/mono_duration10.0_overlap0.8/master.h5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration20.0_overlap0.5/master.h5',
#     # ]

#     print(f"Found {len(h5_files)} H5 files under {root_dir}")
#     for h5_path in h5_files:
#         print(h5_path)
#         # if h5_path in already_corrected:
#         #     continue
#         # patch_one_h5_label_file(
#         #     h5_path,
#         #     old_to_new=old_to_new,
#         #     make_backup=make_backup,
#         # )

#     # print("Done patching all files.")


# if __name__ == "__main__":

#     # already_corrected = [
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration10.0_overlap0.5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/mono_duration10.0_overlap0.5',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration10.0_overlap0.8',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/mono_duration10.0_overlap0.8',
#     #     '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration20.0_overlap0.5',
#     # ]

#     # for path in already_corrected:
#     patch_all_master_h5_under_root(
#         '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/',
#         # '/mnt/data/anphan/AHEAP_data/all_h5_master_files_250hz/bi23_duration20.0_overlap0.8/',
#         old_to_new={0: 1, 1: 0, 2: 2},
#         make_backup=True,
#     )