#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

DEFAULT_CHECKPOINTS=(
    "checkpoints/hybrid_jointwise_mixste_h36m_cpn_one_step_interp_shared_patch3"
)

PARTITION="COOP"
GPU_TYPE=""
NODELIST=""
GPUS=1
CPUS_PER_TASK=8
MEMORY="10G"
TIME_LIMIT="01:00:00"
JOB_NAME="framewise-ablation"
OUTPUT_ROOT="framewise_ablation"
MASTER_LOG_DIR="${SCRIPT_DIR}/slurm_logs/framewise_ablation"
MODULE_PYTORCH="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
MODULE_EINOPS="einops/0.7.0-GCCcore-12.3.0"
MODULE_TIMM="timm/1.0.3-foss-2023a-CUDA-12.1.1"
PYTHON_BIN="python3"

CHECKPOINTS=()
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash run_ablation_framewise_decoder.sh [launcher-options] [-- extra-python-args]

Launcher options:
  --checkpoint-dir <DIR>     Repeatable checkpoint directory. If omitted, uses curated defaults.
  --partition <COOP|FARM>    Slurm partition.
  --nodelist <NODE[,NODE]>   Restrict the job to specific Slurm node names.
  --gpu-type <4090|a100>     GRES GPU type; auto-selected from partition if omitted.
  --gpus <N>                 Number of GPUs to request.
  --cpus <N>                 --cpus-per-task.
  --mem <SIZE>               --mem.
  --time <HH:MM:SS>          --time.
  --job-name <NAME>          Slurm job name.
  --output-root <DIR>        Output root for CSV/TXT metrics.
  --module-pytorch <MODULE>  PyTorch module.
  --module-einops <MODULE>   einops module.
  --module-timm <MODULE>     timm module.
  --python <BIN>             Python executable.
  -h, --help                 Show this help.

Examples:
  bash run_ablation_framewise_decoder.sh
  bash run_ablation_framewise_decoder.sh -- --decoder_mode_override one_step_interp
  bash run_ablation_framewise_decoder.sh --checkpoint-dir checkpoints/foo -- --align_corners_override true
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)
            CHECKPOINTS+=("$2")
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --nodelist)
            NODELIST="$2"
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --module-pytorch)
            MODULE_PYTORCH="$2"
            shift 2
            ;;
        --module-einops)
            MODULE_EINOPS="$2"
            shift 2
            ;;
        --module-timm)
            MODULE_TIMM="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${GPU_TYPE}" ]]; then
    if [[ "${PARTITION}" == "FARM" ]]; then
        GPU_TYPE="a100"
    else
        GPU_TYPE="4090"
    fi
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    CHECKPOINTS=("${DEFAULT_CHECKPOINTS[@]}")
fi

mkdir -p "${MASTER_LOG_DIR}"

PYTHON_ARGS=(
    "--output_root" "${OUTPUT_ROOT}"
)

for checkpoint_dir in "${CHECKPOINTS[@]}"; do
    PYTHON_ARGS+=("--checkpoint_dir" "${checkpoint_dir}")
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submitting frame-wise ablation on ${PARTITION} with gpu:${GPU_TYPE}:${GPUS}, cpu:${CPUS_PER_TASK}, mem:${MEMORY}, time:${TIME_LIMIT}"
    SBATCH_CMD=(
        sbatch
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --gres="gpu:${GPU_TYPE}:${GPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --output="${MASTER_LOG_DIR}/%x.%j.out" \
        "$0" \
        --partition "${PARTITION}" \
        --gpu-type "${GPU_TYPE}" \
        --gpus "${GPUS}" \
        --cpus "${CPUS_PER_TASK}" \
        --mem "${MEMORY}" \
        --time "${TIME_LIMIT}" \
        --job-name "${JOB_NAME}" \
        --output-root "${OUTPUT_ROOT}" \
        --module-pytorch "${MODULE_PYTORCH}" \
        --module-einops "${MODULE_EINOPS}" \
        --module-timm "${MODULE_TIMM}" \
        --python "${PYTHON_BIN}" \
    )
    if [[ -n "${NODELIST}" ]]; then
        SBATCH_CMD+=(--nodelist="${NODELIST}")
    fi
    for checkpoint_dir in "${CHECKPOINTS[@]}"; do
        SBATCH_CMD+=(--checkpoint-dir "${checkpoint_dir}")
    done
    SBATCH_CMD+=(-- "${EXTRA_ARGS[@]}")
    "${SBATCH_CMD[@]}"
    exit 0
fi

source /etc/profile
module purge
module load "${MODULE_PYTORCH}"
module load "${MODULE_EINOPS}"
module load "${MODULE_TIMM}"

export OMP_NUM_THREADS="${CPUS_PER_TASK}"

echo "Running frame-wise decoder ablation with ${PYTHON_BIN} on $(hostname)"
"${PYTHON_BIN}" ablation_framewise_decoder.py "${PYTHON_ARGS[@]}" "${EXTRA_ARGS[@]}"
