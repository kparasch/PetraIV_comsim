import petra_SC as pSC
from pySC.utils import logging_tools
import numpy as np
import matplotlib.pyplot as plt

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=1)
args = argparser.parse_args()

LOGGER = logging_tools.get_logger(__name__)

seed = args.seed
np.random.seed(seed)

Pem = pSC.PetraErrorModel()
p424 = 'p4_H6BA_v4_2_4.mat'
SC = pSC.register_petra_stuff(p424, Pem)
pSC.number_of_elements(SC)

SC = pSC.apply_errors_and_init(SC, plot=True)
knobs = pSC.PetraKnobs(SC.RING)
SC.plot = False

pSC._save_and_check_repr(SC.RING, f'initial_errors_seed{seed}.repr')

plt.show()