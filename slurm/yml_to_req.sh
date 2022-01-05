#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara

#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --job-name=YML_TO_REQ
#SBATCH --mail-user=cecilianlv@icloud.com # adjust this to match your email address
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j-YML_TO_REQ.out

cd $SCRATCH

# Load your modules as before
module load CCEnv arch/avx512 StdEnv/2020

module load gcc openmm python/3.7 openmpi 

# Generate your virtual environment in $SLURM_TMPDIR
virtualenv --no-download ${SLURM_TMPDIR}/my_env && source ${SLURM_TMPDIR}/my_env/bin/activate

# Install alphafold and dependencies
pip install --no-index ruamel

python $SCRATCH/ProteinGAN/yml_to_requirements.py