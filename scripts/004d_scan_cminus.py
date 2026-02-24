import argparse
from pySC import SimulatedCommissioning, ResponseMatrix
from pySC.utils import rdt
from pySC.core.control import IndivControl
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
    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_03_seed{seed}.json')
    SC.injection.n_particles = 1


    # define knob
    SC.tuning.c_minus.controls = [name + '/A2L' for name in SC.magnet_arrays['skew_quads']]
    SC.import_knob('model_RM/c_minus_knobs.json')

    # do the scan
    SC.lattice.omp_num_threads=args.threads
    SC.injection.n_particles=100

    sp_cm = np.linspace(-0.2,0.2,21)
    scan_turns = 500
    injef = np.zeros_like(sp_cm)
    for ii, sp in enumerate(sp_cm):
        SC.magnet_settings.set('c_minus_real', sp)
        injef[ii] = SC.tuning.injection_efficiency(n_turns=scan_turns)[-1]
        print(f'c_minus_real = {sp}, injef={injef[ii]*100}%')

    best_sp = sp_cm[np.argmax(injef)]
    SC.magnet_settings.set('c_minus_real', best_sp)
    print(f"Setting c_minus_real to {best_sp}")

    sp_cm = np.linspace(-0.2,0.2,21)
    injef = np.zeros_like(sp_cm)
    for ii, sp in enumerate(sp_cm):
        SC.magnet_settings.set('c_minus_imag', sp)
        injef[ii] = SC.tuning.injection_efficiency(n_turns=scan_turns)[-1]
        print(f'c_minus_imag = {sp}, injef={injef[ii]*100}%')

    best_sp = sp_cm[np.argmax(injef)]
    SC.magnet_settings.set('c_minus_imag', best_sp)
    print(f"Setting c_minus_imag to {best_sp}")

    SC.injection.n_particles = 512
    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    SC.to_json(f'data/Seeds/pySC_petra4_04_seed{seed}.json')
    np.savez(f"data/Seeds/injection_efficiency_before_orbit{seed}", injef)