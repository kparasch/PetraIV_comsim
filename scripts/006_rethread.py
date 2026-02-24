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

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_05_seed{seed}.json')

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None

    # reset tune quadrupoles
    quads1 = SC.tuning.tune.tune_quad_controls_1
    quads2 = SC.tuning.tune.tune_quad_controls_2
    quad_ref = SC.design_magnet_settings.get_many(quads1 + quads2)
    SC.magnet_settings.set_many(quad_ref)

    # reset correctors
    for corr in SC.tuning.CORR:
        SC.magnet_settings.set(corr, 0.)

    #thread 
    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
        SC.tuning.wiggle_last_corrector(max_steps=100, max_sp=500e-6)
    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=1)
    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=1, gain=0.5)
    for _ in range(20):
        SC.tuning.correct_injection(parameter=50, n_reps=10, n_turns=2, gain=0.2)

    SC.injection.n_particles = 512
    SC.lattice.omp_num_threads = threads

    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    np.savez(f"data/Seeds/injection_efficiency_second_{seed}", injef)

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None
    SC.to_json(f'data/Seeds/pySC_petra4_06_seed{seed}.json')
