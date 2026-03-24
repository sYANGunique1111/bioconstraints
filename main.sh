#!/bin/bash
set -euo pipefail

# Training launcher for Slurm execution on this cluster.
#
# Examples:
#   bash main.sh
#   bash main.sh --partition COOP --gpus 1
#   bash main.sh --partition FARM --gpus 2 --time 12:00:00
#
# Pass extra training args to main.py after `--`, e.g.:
#   bash main.sh --partition COOP -- --epochs 200

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}"

# Training defaults
KEYPOINTS="cpn_ft_h36m_dbb"
MODEL="hybrid_jointwise_mixste"
CHECKPOINT_DIR="hybrid_jointwise_mixste_h36m_cpn_two_step_mix_shared_patch3"
DECODER_MODE="two_step_mix"
EMBED_MODE="shared"
PATCH_SIZE="3"

# Runtime defaults (conservative and server-friendly)
# COOP: gpu:4090, 110 GB RAM/node
# FARM: gpu:a100, 366 GB RAM/node
PARTITION="FARM"
GPU_TYPE=""          # Auto-set from partition if not provided.
NODELIST=""
GPUS=1               # Number of GPUs requested. Also used as world_size.
CPUS_PER_TASK=8      # CPU threads available to each training process.
MEMORY="20G"         # Total memory for the job.
TIME_LIMIT="20:00:00"
JOB_NAME="bioconstraints-train"
SLURM_LOG_DIR="${SCRIPT_DIR}/checkpoints/${CHECKPOINT_DIR}"
MASTER_PORT="8500"
MODULE_PYTORCH="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
MODULE_EINOPS="einops/0.7.0-GCCcore-12.3.0"
MODULE_TIMM="timm/1.0.3-foss-2023a-CUDA-12.1.1"
PYTHON_BIN="python3"

EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash main.sh [launcher-options] [-- extra-main.py-args]

Launcher options:
  --partition <COOP|FARM>    Slurm partition.
  --nodelist <NODE[,NODE]>   Restrict the job to specific Slurm node names.
  --gpu-type <4090|a100>     GRES GPU type; auto-selected from partition if omitted.
  --gpus <N>                 Number of GPUs to request (also sets --world_size).
  --cpus <N>                 --cpus-per-task.
  --mem <SIZE>               --mem (examples: 48G, 120G).
  --time <HH:MM:SS>          --time.
  --job-name <NAME>          Slurm job name.
  --master-port <PORT>       DDP master port.
  --module-pytorch <MODULE>  PyTorch module to load.
  --module-einops <MODULE>   einops module required by models/mixste.py.
  --module-timm <MODULE>     timm module required by models/mixste.py.
  --python <BIN>             Python executable (default: python3).
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
        --master-port)
            MASTER_PORT="$2"
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

if [[ ! -d "${SCRIPT_DIR}/checkpoints" ]]; then
    mkdir -p "${SCRIPT_DIR}/checkpoints"
fi
if [[ ! -d "${SLURM_LOG_DIR}" ]]; then
    mkdir -p "${SLURM_LOG_DIR}"
fi

TRAIN_ARGS=(
    "--epochs" "150"
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "${KEYPOINTS}"
    "--model" "${MODEL}"
    "--embed_dim" "512"
    "--depth" "7"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    "--patch_size" "${PATCH_SIZE}"
    "--decoder_mode" "${DECODER_MODE}"
    "--embed_mode" "${EMBED_MODE}"
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    "--checkpoint" "checkpoints/${CHECKPOINT_DIR}"
    "--world_size" "${GPUS}"
    "--master_port" "${MASTER_PORT}"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"
    "--nolog"
)

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submitting Slurm job on ${PARTITION} with gpu:${GPU_TYPE}:${GPUS}, cpu:${CPUS_PER_TASK}, mem:${MEMORY}, time:${TIME_LIMIT}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        ${NODELIST:+--nodelist="${NODELIST}"} \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --gres="gpu:${GPU_TYPE}:${GPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --output="${SLURM_LOG_DIR}/%x.%j.out" \
        "$0" \
        --partition "${PARTITION}" \
        ${NODELIST:+--nodelist "${NODELIST}"} \
        --gpu-type "${GPU_TYPE}" \
        --gpus "${GPUS}" \
        --cpus "${CPUS_PER_TASK}" \
        --mem "${MEMORY}" \
        --time "${TIME_LIMIT}" \
        --job-name "${JOB_NAME}" \
        --master-port "${MASTER_PORT}" \
        --module-pytorch "${MODULE_PYTORCH}" \
        --module-einops "${MODULE_EINOPS}" \
        --module-timm "${MODULE_TIMM}" \
        --python "${PYTHON_BIN}" \
        -- "${EXTRA_ARGS[@]}"
    exit 0
fi

if ! type module >/dev/null 2>&1; then
    # Required for non-login shells where Lmod is not initialized automatically.
    source /etc/profile
fi

module purge
module load "${MODULE_PYTORCH}"
module load "${MODULE_EINOPS}"
module load "${MODULE_TIMM}"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${CPUS_PER_TASK}"

echo "Running training with ${PYTHON_BIN} on $(hostname)"
"${PYTHON_BIN}" main.py "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
