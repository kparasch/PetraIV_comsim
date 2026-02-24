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
    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_04_seed{seed}.json')
    SC.injection.n_particles = 1

    # SC.tuning.tune.correct(n_iter=1, gain=1., measurement_method='cheat_with_integer')
    # SC.tuning.c_minus.correct(gain=0.5)
    n_iter = 20
    c_iter = 3
    gain = 0.5
    for ii in range(n_iter):
        logger.info(f"Iteration {ii+1}/{n_iter}:")
        SC.tuning.tune.correct(n_iter=1, gain=gain, measurement_method='cheat_with_integer')
        # SC.lattice.use_orbit_guess = False
        for _ in range(c_iter):
            SC.tuning.c_minus.correct(gain=gain)

        # dfreq = SC.tuning.fit_dispersive_orbit()
        # logger.info(f'Frequency correction for dispersive orbit {dfreq:.1f}Hz.')
        # SC.rf_settings.main.set_frequency(SC.rf_settings.main.frequency - dfreq)
        if ii > 5:
            SC.tuning.correct_orbit(parameter=20, gain=0.5)

    SC.injection.n_particles = 512
    SC.lattice.omp_num_threads = args.threads
    SC.lattice.use_orbit_guess = False

    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    np.savez(f"data/Seeds/injection_efficiency_first_{seed}", injef)

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None
    SC.to_json(f'data/Seeds/pySC_petra4_04b_seed{seed}.json')
