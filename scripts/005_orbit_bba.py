import argparse
from pySC import SimulatedCommissioning
import logging


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    argparser.add_argument('--n_iter', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_04b_seed{seed}.json')

    SC.tuning.generate_orbit_bba_config()
    SC.injection.n_particles = 1
    for _ in range(args.n_iter):
        if args.threads > 1:
            SC.tuning.do_parallel_orbit_bba(omp_num_threads=args.threads)
        else:
            SC.tuning.do_orbit_bba()

        # n_iter_cor = 8
        # gain = 0.5
        # for ii in range(n_iter_cor):
        #     logger.info(f"Iteration {ii+1}/{n_iter_cor}:")
        #     SC.tuning.correct_orbit(parameter=20, gain=gain)
        #     dfreq = SC.tuning.fit_dispersive_orbit()
        #     logger.info(f'Frequency correction for dispersive orbit {dfreq:.1f}Hz.')
        #     SC.rf_settings.main.set_frequency(SC.rf_settings.main.frequency - gain*dfreq)
        #     SC.tuning.tune.correct(n_iter=1, gain=gain, measurement_method='cheat_with_integer')
        #     SC.tuning.c_minus.correct(n_iter=1, gain=gain)


    SC.to_json(f'data/Seeds/pySC_petra4_05_seed{seed}.json')
