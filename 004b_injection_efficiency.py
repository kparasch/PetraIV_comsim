import petra_SC as pSC
from pySC.utils import logging_tools
from pySC.core.beam import beam_transmission
from pySC.correction.bba import orbit_bba
from pySC.lattice_properties.response_model import SCgetModelRM
import numpy as np
import matplotlib.pyplot as plt

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=1)
argparser.add_argument('--repr', type=str, default=None)
args = argparser.parse_args()

LOGGER = logging_tools.get_logger(__name__)

seed = args.seed
np.random.seed(seed)

if args.repr is None:
    repr_file = f'after_RF_setup_seed{seed}.repr'
else:
    repr_file = args.repr

Pem = pSC.PetraErrorModel()
p424 = 'p4_H6BA_v4_3_3.mat'
SC = pSC.register_petra_stuff(p424, Pem)
pSC.number_of_elements(SC)
knobs = pSC.PetraKnobs(SC.RING)
SC.plot = False
SC.RING = pSC._load_repr(repr_file)
#SC.RING = pSC._load_repr(f'after_RF_setup_seed{seed}.repr')
SC.RING = pSC.fix_apertures(SC.RING)

SC.INJ.Z0 = np.zeros(6)
SC.INJ.trackMode = 'TBT'
max_t, frac = beam_transmission(SC, nParticles=200, nTurns=1000, plot=False)
print(f'Injection efficiency is {frac[-1]*100}%.')
np.savez(f"injection_efficiency_before_orbit{seed}", frac)
#np.savez(f"ORM_{repr_file}", RM)
#pSC.calculate_and_save_response_matrices(SC, [f'ORM1_{repr_file}', f'ORM2_{repr_file}'])
