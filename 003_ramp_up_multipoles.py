import petra_SC as pSC
from pySC.utils import logging_tools, sc_tools
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
knobs = pSC.PetraKnobs(SC.RING)
SC.plot = False
#SC.RING = pSC._load_repr(f'after_threading_seed{seed}.repr')
SC.RING = pSC._load_repr(f'after_trajBBA_seed{seed}.repr')
SC.RING = pSC.fix_apertures(SC.RING)

SC.set_cm_setpoints(SC.ORD.HCM, 0.0, False)
SC.set_cm_setpoints(SC.ORD.VCM, 0.0, True) 
SC.INJ.Z0 = np.zeros(6)
# SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^QF|^QD'), 1.0, False, 1, method='rel')
SC = pSC.beam_threading(SC, Pem, run_rm=False)
LOGGER.info("Beam re-threading finished")
# SC.RING = switch_rf(SC.RING, SC.ORD.RF, False)
SC.INJ.nShots = 3

SC = pSC.run_multipoles(SC, Pem, run_rm=False, ramp_multipoles=True, nsteps=10)
SC.INJ.nShots = 1

pSC._save_and_check_repr(SC.RING, f'after_multipole_rampup_seed{seed}.repr')

plt.show()