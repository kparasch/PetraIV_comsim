#!/bin/bash
#SBATCH --time=0-24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=allcpu
#SBATCH --job-name=pySC

seed=$1

unset LD_PRELOAD                     # useful on max-display nodes, harmless on others
source /etc/profile.d/modules.sh     # make the module command available

source /data/dust/user/parascho/miniforge3/bin/activate base

comsim=/home/parascho/dust/003_newSC

threads=8

pwd
cd $comsim

python scripts/000_generate_SC.py --seed $seed
python scripts/001_thread.py --seed $seed
python scripts/002_tBBA.py --seed $seed --threads $threads --n_iter 2
python scripts/003_ramp_multipoles.py --seed $seed --threads $threads
python scripts/004_scan_tune_phase.py --seed $seed --threads $threads
python scripts/004x_xopt_tune_coupling.py --seed $seed --threads $threads
python scripts/005_orbit_bba.py --seed $seed --threads $threads


# spare
# python scripts/006_clean_corrs.py --seed $seed
# python scripts/006_rethread.py --seed $seed --threads $threads
# python scripts/007_tune_orbit.py --seed $seed --threads $threads
