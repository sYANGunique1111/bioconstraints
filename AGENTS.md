# AGENTS.md

## Python Execution Rule

- Before running any `python3` command, activate Conda base:
  - `source /FARM/syangb/miniconda3/etc/profile.d/conda.sh && conda activate base`

## Server Module Loading (Current Cluster)

- This server uses environment modules for the PyTorch stack; Conda base alone is not enough for `torch`/`einops`/`timm`.
- For any Slurm-side Python execution that needs ML dependencies, initialize modules exactly like `main.sh`:
  - `source /etc/profile`
  - `module purge`
  - `module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`
  - `module load einops/0.7.0-GCCcore-12.3.0`
  - `module load timm/1.0.3-foss-2023a-CUDA-12.1.1`
- General execution pattern:
  - Use `srun ... bash -lc 'source /etc/profile && module purge && module load ... && python3 ...'`
  - Or use `bash main.sh --slurm ...` so module setup is handled automatically.
