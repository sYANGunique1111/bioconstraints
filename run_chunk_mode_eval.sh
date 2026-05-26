#!/bin/bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

PARTITION="COOP"
GPU_TYPE=""
NODELIST=""
GPUS=1
CPUS_PER_TASK=8
MEMORY="20G"
TIME_LIMIT="04:00:00"
JOB_NAME="chunk-mode-eval"
SLURM_LOG_DIR="${SCRIPT_DIR}/checkpoints/chunk_mode_eval"
PYTHON_BIN="python3"
MODULE_PYTORCH="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
MODULE_EINOPS="einops/0.7.0-GCCcore-12.3.0"
MODULE_TIMM="timm/1.0.3-foss-2023a-CUDA-12.1.1"
DATASET_ROOT="${H36M_DATASET_ROOT:-/FARM/syangb/data/h36m}"
OUTPUT_CSV="checkpoints/chunk_mode_eval/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_matrix.csv"
DEFAULT_OUTPUT_CSV="${OUTPUT_CSV}"
STRIDE_CROSS_EVAL=false

EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash run_chunk_mode_eval.sh [launcher-options] [-- extra-eval-args]

Launcher options:
  --partition <COOP|FARM>    Slurm partition.
  --nodelist <NODE[,NODE]>   Restrict the job to specific Slurm node names.
  --gpu-type <4090|a100>     GRES GPU type; auto-selected from partition if omitted.
  --gpus <N>                 Number of GPUs to request.
  --cpus <N>                 --cpus-per-task.
  --mem <SIZE>               --mem (examples: 20G, 48G).
  --time <HH:MM:SS>          --time.
  --job-name <NAME>          Slurm job name.
  --dataset-root <PATH>      Override H36M dataset root for evaluation.
  --output-csv <PATH>        Output CSV path.
  --python <BIN>             Python executable (default: python3).
  --module-pytorch <MODULE>  PyTorch module to load.
  --module-einops <MODULE>   einops module to load.
  --module-timm <MODULE>     timm module to load.
  --stride-cross-eval        Evaluate the stride-trained checkpoint on drop/center_pad/tail_pad.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
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
        --stride-cross-eval)
            STRIDE_CROSS_EVAL=true
            shift
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

mkdir -p "${SLURM_LOG_DIR}"

if [[ "${STRIDE_CROSS_EVAL}" == "true" ]]; then
    if [[ "${OUTPUT_CSV}" == "${DEFAULT_OUTPUT_CSV}" ]]; then
        OUTPUT_CSV="checkpoints/chunk_mode_eval/hot_mixste_chunked_layer3_token81_feature_interp_attn_identifier_embed_stride_cross_eval.csv"
    fi
fi

EVAL_ARGS=(
    "--dataset_root" "${DATASET_ROOT}"
    "--output_csv" "${OUTPUT_CSV}"
)

if [[ "${STRIDE_CROSS_EVAL}" == "true" ]]; then
    EVAL_ARGS+=("--stride_cross_eval")
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submitting Slurm job on ${PARTITION} with gpu:${GPU_TYPE}:${GPUS}, cpu:${CPUS_PER_TASK}, mem:${MEMORY}, time:${TIME_LIMIT}"
    SBATCH_ARGS=(
        --job-name="${JOB_NAME}"
        --partition="${PARTITION}"
        --nodes=1
        --ntasks=1
        --cpus-per-task="${CPUS_PER_TASK}"
        --gres="gpu:${GPU_TYPE}:${GPUS}"
        --mem="${MEMORY}"
        --time="${TIME_LIMIT}"
        --output="${SLURM_LOG_DIR}/%x.%j.out"
        "$0"
        --partition "${PARTITION}"
        --gpu-type "${GPU_TYPE}"
        --gpus "${GPUS}"
        --cpus "${CPUS_PER_TASK}"
        --mem "${MEMORY}"
        --time "${TIME_LIMIT}"
        --job-name "${JOB_NAME}"
        --dataset-root "${DATASET_ROOT}"
        --output-csv "${OUTPUT_CSV}"
        --python "${PYTHON_BIN}"
        --module-pytorch "${MODULE_PYTORCH}"
        --module-einops "${MODULE_EINOPS}"
        --module-timm "${MODULE_TIMM}"
    )
    if [[ -n "${NODELIST}" ]]; then
        SBATCH_ARGS+=(--nodelist="${NODELIST}")
    fi
    if [[ "${STRIDE_CROSS_EVAL}" == "true" ]]; then
        SBATCH_ARGS+=(--stride-cross-eval)
    fi
    SBATCH_ARGS+=(-- "${EXTRA_ARGS[@]}")
    sbatch "${SBATCH_ARGS[@]}"
    exit 0
fi

source /FARM/syangb/miniconda3/etc/profile.d/conda.sh
conda activate base
set +u
source /etc/profile
set -u
module purge
module load "${MODULE_PYTORCH}"
module load "${MODULE_EINOPS}"
module load "${MODULE_TIMM}"

export OMP_NUM_THREADS="${CPUS_PER_TASK}"
export H36M_DATASET_ROOT="${DATASET_ROOT}"

echo "Running chunk-mode evaluation with ${PYTHON_BIN} on $(hostname)"
"${PYTHON_BIN}" evaluate_chunk_mode_matrix.py "${EVAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
