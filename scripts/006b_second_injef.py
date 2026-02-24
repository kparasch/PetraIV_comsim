
from pySC.core.new_simulated_commissioning import SimulatedCommissioning
import numpy as np
import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed
    threads = args.threads if args.threads > 1 else None


    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_06_seed{seed}.json')

    SC.injection.n_particles = 512
    SC.lattice.omp_num_threads = threads

    injef = SC.tuning.injection_efficiency(n_turns=2000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    np.savez(f"data/Seeds/injection_efficiency_second_{seed}", injef)