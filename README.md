# PetraIV_comsim

example:
```
#!/bin/bash

seed=1
python 000_generate_seed.py --seed $seed
python 001_threading.py --seed $seed
python 002_trajectory_bba.py --seed $seed
python 003_ramp_up_multipoles.py --seed $seed --alpha 50
python 003_ramp_up_multipoles.py --seed $seed --alpha 40
python 003_ramp_up_multipoles.py --seed $seed --alpha 30
python 003_ramp_up_multipoles.py --seed $seed --alpha 20
python 004_setup_rf.py --seed $seed
python 005_orbit_response_matrix.py --seed $seed
python 005b_fix_tune_orbit.py --seed $seed
python 006_orbit_bba.py --seed $seed
python 005c_fix_tune_orbit.py --seed $seed
```
