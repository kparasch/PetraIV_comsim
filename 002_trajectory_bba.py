import petra_SC as pSC
from pySC.utils import logging_tools
import numpy as np
import matplotlib.pyplot as plt

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=1)
argparser.add_argument('--plot', action='store_true')
args = argparser.parse_args()

LOGGER = logging_tools.get_logger(__name__)

seed = args.seed
np.random.seed(seed)

Pem = pSC.PetraErrorModel()
p424 = 'p4_H6BA_v4_3_3.mat'
SC = pSC.register_petra_stuff(p424, Pem)
pSC.number_of_elements(SC)
knobs = pSC.PetraKnobs(SC.RING)
SC.plot = False
SC.RING = pSC._load_repr(f'after_threading_seed{seed}.repr')
SC.RING = pSC.fix_apertures(SC.RING)

SC.INJ.Z0 = np.zeros(6)
SC.RING = pSC.switch_rf(SC.RING, SC.ORD.RF, False)
SC.INJ.nTurns = 2
SC.INJ.nParticles = 1
SC.INJ.nShots = 1
SC = pSC.perform_trajectory_bba(SC, knobs, n_bpms=None)

pSC._save_and_check_repr(SC.RING, f'after_trajBBA_seed{seed}.repr')

plt.show(block=args.plot)