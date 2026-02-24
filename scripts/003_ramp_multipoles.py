from pySC import SimulatedCommissioning
import numpy as np
import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed
    threads = args.threads if args.threads > 1 else None

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_02_seed{seed}.json')

    max_scale = 0.97
    SC.injection.n_particles = 200
    SC.lattice.omp_num_threads = threads

    for ii, tr in enumerate(SC.tuning.injection_efficiency(n_turns=10)[::2]):
        print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    n_steps = 10
    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None
    for ii in range(n_steps+1):
        scale_to_set = min(max_scale, ii / n_steps)
        SC.tuning.set_multipole_scale(scale_to_set)
        for _ in range(5):
            SC.tuning.correct_injection(method='svd_cutoff', parameter=1e-2, n_reps=1, n_turns=2, gain=0.2)

    SC.injection.n_particles = 200
    SC.lattice.omp_num_threads = threads

    for ii, tr in enumerate(SC.tuning.injection_efficiency(n_turns=10)):
        print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None
    SC.to_json(f'data/Seeds/pySC_petra4_03_seed{seed}.json')