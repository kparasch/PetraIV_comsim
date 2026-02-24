import argparse
from pySC import SimulatedCommissioning
import matplotlib.pyplot as plt
import numpy as np



argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=1)
args = argparser.parse_args()

seed = args.seed

SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_00_seed{seed}.json')


SC.tuning.set_multipole_scale(scale=0)

SC.injection.n_particles = 1
for _ in range(20):
    SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
    SC.tuning.wiggle_last_corrector(max_steps=100, max_sp=500e-6)
for _ in range(20):
    SC.tuning.tune.correct(n_iter=1, gain=0.5, measurement_method='first_turn')
    SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
for _ in range(20):
    SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=0.5)
for _ in range(20):
    SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=2, gain=0.2)

SC.to_json(f'data/Seeds/pySC_petra4_01_seed{seed}.json')