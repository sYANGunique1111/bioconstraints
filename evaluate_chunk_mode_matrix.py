"""
Run a checkpoint-by-evaluation-mode sweep for sequence chunking experiments.

Default sweep:
  - checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_drop
  - checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_center
  - checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_tail

Each checkpoint is evaluated under:
  - drop
  - center_pad
  - tail_pad
"""

import argparse
import csv
import os

from evaluate import evaluate_checkpoint


DEFAULT_CHECKPOINTS = [
    "checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_drop",
    "checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_center",
    "checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_tail",
]
DEFAULT_MODES = ["drop", "center_pad", "tail_pad"]
STRIDE_CHECKPOINT = "checkpoints/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_stride"


def infer_train_mode_label(checkpoint_path):
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if name.endswith("_drop"):
        return "drop"
    if name.endswith("_center"):
        return "center_pad"
    if name.endswith("_tail"):
        return "tail_pad"
    if name.endswith("_stride"):
        return "stride"
    return "unknown"


def write_csv(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint_name",
                "checkpoint_path",
                "train_chunk_mode",
                "eval_chunk_mode",
                "mpjpe_mm",
                "pa_mpjpe_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\n" + "=" * 88)
    print(f"{'Checkpoint':<68} {'Eval mode':<12} {'MPJPE':>8} {'PA-MPJPE':>12}")
    print("=" * 88)
    for row in rows:
        print(
            f"{row['checkpoint_name']:<68} "
            f"{row['eval_chunk_mode']:<12} "
            f"{row['mpjpe_mm']:>8.3f} "
            f"{row['pa_mpjpe_mm']:>12.3f}"
        )
    print("=" * 88 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints across multiple chunking modes")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint directories to evaluate",
    )
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
        help="Evaluation chunking modes to sweep",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional DataLoader batch size for evaluation",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional H36M dataset root; defaults to H36M_DATASET_ROOT or known local paths",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="checkpoints/chunk_mode_eval/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_matrix.csv",
        help="Where to write the CSV summary",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip per-action breakdown and evaluate all test actions together",
    )
    parser.add_argument(
        "--stride_cross_eval",
        action="store_true",
        help="Evaluate the stride-trained checkpoint on drop, center_pad, and tail_pad only",
    )
    args = parser.parse_args()

    if args.stride_cross_eval:
        args.checkpoints = [STRIDE_CHECKPOINT]
        args.eval_modes = DEFAULT_MODES

    rows = []
    for checkpoint_dir in args.checkpoints:
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        train_chunk_mode = infer_train_mode_label(checkpoint_dir)
        for eval_mode in args.eval_modes:
            print(f"Running evaluation: checkpoint={checkpoint_name}, eval_mode={eval_mode}")
            result = evaluate_checkpoint(
                checkpoint_dir=checkpoint_dir,
                batch_size=args.batch_size,
                sequence_chunk_mode=eval_mode,
                dataset_root=args.dataset_root,
                action_wise=not args.aggregate_only,
            )
            rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_path": checkpoint_dir,
                    "train_chunk_mode": train_chunk_mode,
                    "eval_chunk_mode": eval_mode,
                    "mpjpe_mm": result["mpjpe_mm"],
                    "pa_mpjpe_mm": result["pa_mpjpe_mm"],
                }
            )

    write_csv(rows, args.output_csv)
    print_summary(rows)
    print(f"Saved CSV summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
