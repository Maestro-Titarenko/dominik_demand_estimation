#!/bin/bash
#SBATCH --job-name=nevo_blp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=<onyen>@email.unc.edu

# Nevo (2000)-style BLP estimation on Longleaf (benchmark defaults of 03_blp_nevo.py).
# Submit from the project root:
#     sbatch code/longleaf/run_nevo.sl                                   # full sample benchmark
#     sbatch --export=ALL,EVERY=13 --mem=16g code/longleaf/run_nevo.sl   # quarterly subsample
#     sbatch --export=ALL,EXTRA="--mw-quantile 0.99" code/longleaf/run_nevo.sl   # robustness
# Variables: FE (default store_week), EVERY (default 1 = full sample), EXTRA (other options).
# Benchmark full sample: 64 min, 37 GB.

set -euo pipefail
FE=${FE:-store_week}
EVERY=${EVERY:-1}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output
module purge
module load python/3.12.4
source .venv/bin/activate

# Serial on purpose: pyblp.parallel was far slower than 1 process on this problem
# (many small markets, so inter-process data transfer dominates).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname): FE=$FE EVERY=$EVERY EXTRA=${EXTRA:-} CPUS=$SLURM_CPUS_PER_TASK"
python code/03_blp_nevo.py --every "$EVERY" --fe "$FE" --processes "$SLURM_CPUS_PER_TASK" ${EXTRA:-}
