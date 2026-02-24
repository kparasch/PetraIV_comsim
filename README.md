# PetraIV_comsim

example:
```
#!/bin/bash

python scripts/000_generate_SC.py --seed $seed
python scripts/001_thread.py --seed $seed
python scripts/002_tBBA.py --seed $seed --threads $threads --n_iter 2
python scripts/003_ramp_multipoles.py --seed $seed --threads $threads
python scripts/004_scan_tune_phase.py --seed $seed --threads $threads
python scripts/004x_xopt_tune_coupling.py --seed $seed --threads $threads
python scripts/005_orbit_bba.py --seed $seed --threads $threads
```
