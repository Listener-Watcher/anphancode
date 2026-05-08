import argparse
from caueeg_main import run_caueeg_ladder
from caueeg_ladders import LADDER_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--ladder", required=True)
parser.add_argument("--dataset-path", required=True)
parser.add_argument("--h5-path", required=True)
parser.add_argument("--task", default="dementia")
parser.add_argument("--output-root", required=True)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--match", type=str, default=None)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

builder = LADDER_REGISTRY[args.ladder]
ladder = builder(
    dataset_path=args.dataset_path,
    h5_path=args.h5_path,
    task=args.task,
    output_root=args.output_root,
)

if args.match:
    ladder = [spec for spec in ladder if args.match.lower() in spec.name.lower()]

ladder = ladder[args.start:args.end]

for i, spec in enumerate(ladder):
    print(f"{i:02d} - {spec.name}")

if not args.dry_run:
    run_caueeg_ladder(ladder)