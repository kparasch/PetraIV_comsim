import argparse
from pySC import SimulatedCommissioning
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed
    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_06_seed{seed}.json')
    SC.injection.n_particles = 1

    n_iter = 20
    for ii in range(n_iter):
        logger.info(f"Iteration {ii+1}/{n_iter}:")
        SC.tuning.tune.correct(n_iter=1, gain=1., measurement_method='cheat_with_integer')

        # dfreq = SC.tuning.fit_dispersive_orbit()
        # logger.info(f'Frequency correction for dispersive orbit {dfreq:.1f}Hz.')
        # SC.rf_settings.main.set_frequency(SC.rf_settings.main.frequency - dfreq)

        SC.tuning.correct_orbit(parameter=50, gain=0.3)

    SC.injection.n_particles = 512
    SC.lattice.omp_num_threads = args.threads

    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    np.savez(f"data/Seeds/injection_efficiency_third_{seed}", injef)

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None
    SC.to_json(f'data/Seeds/pySC_petra4_07_seed{seed}.json')
