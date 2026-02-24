import argparse
from pySC import SimulatedCommissioning
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    argparser.add_argument('--n_iter', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_01_seed{seed}.json')

    SC.tuning.generate_trajectory_bba_config(max_modulation=1000e-6, max_dx_at_bpm=1.5e-3, max_ncorr_index=10)
    names = SC.bpm_system.names[2:]
    SC.injection.n_particles = 1
    for _ in range(args.n_iter):
        if args.threads > 1:
            SC.tuning.do_parallel_trajectory_bba(bpm_names=names, omp_num_threads=args.threads)
        else:
            SC.tuning.do_trajectory_bba(bpm_names=names)

        # re-thread
        for corr in SC.tuning.CORR:
            SC.magnet_settings.set(corr, 0)

        for _ in range(20):
            SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
            SC.tuning.wiggle_last_corrector(max_steps=100, max_sp=500e-6)
        for _ in range(20):
            SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
        for _ in range(40):
            SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=2, gain=0.2)

    # reset tune quads
    SC.magnet_settings.set(SC.tuning.tune.knob_qx, 0)
    SC.magnet_settings.set(SC.tuning.tune.knob_qy, 0)

    # re-thread
    for corr in SC.tuning.CORR:
        SC.magnet_settings.set(corr, 0)

    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
        SC.tuning.wiggle_last_corrector(max_steps=100, max_sp=500e-6)
    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
    for _ in range(40):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=2, gain=0.2)


    SC.to_json(f'data/Seeds/pySC_petra4_02_seed{seed}.json')
