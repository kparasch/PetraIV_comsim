import petra_SC as pSC
from pySC.utils import logging_tools
from pySC.correction.bba import orbit_bba
from pySC.core.beam import bpm_reading
from pySC.lattice_properties.response_model import SCgetModelRM
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
SC.RING = pSC._load_repr(f'after_tune_orbit_seed{seed}.repr')
SC.RING = pSC.fix_apertures(SC.RING)

SC.INJ.Z0 = np.zeros(6)
SC.INJ.trackMode = 'ORB'

ORM = np.load(f'ORM_ideal.npz')['arr_0']

mag_ords = np.tile(knobs.bba_sextupoles, (2, 1))
bpm_ords = np.tile(knobs.bpms_disp_bump, (2, 1))
quad_is_skew = True

n_k1_steps = 5
max_dk1 = 20e-6
n_k2_steps = 2
max_dk2 = 5e-2

SC.orbits = []
SC.bps = []
SC, bba_offsets_skew, bba_offset_errors_skew = orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew=quad_is_skew,
                                                         n_k1_steps=n_k1_steps, max_dk1=max_dk1,
                                                         n_k2_steps=n_k2_steps, max_dk2=max_dk2, RM=ORM,
                                                         plot_results=True)

mag_ords = np.tile(knobs.bba_quads, (2, 1))
bpm_ords = np.tile(knobs.bpms_no_disp, (2, 1))
quad_is_skew = False
max_dk2 = 10e-2

SC, bba_offsets_norm, bba_offset_errors_norm = orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew=quad_is_skew,
                                                         n_k1_steps=n_k1_steps, max_dk1=max_dk1,
                                                         n_k2_steps=n_k2_steps, max_dk2=max_dk2, RM=ORM,
                                                         plot_results=True)




# SC, bba_offsets_skew, bba_offset_errors_skew = orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew,
#                                                          n_k1_steps, max_dk1, n_k2_steps, max_dk2, RM=RM2,
#                                                          plot_results=True)
# 
# mag_ords = np.tile(knobs.bba_quads, (2, 1))
# bpm_ords = np.tile(knobs.bpms_no_disp, (2, 1))
# quad_is_skew = False
# 
# SC, bba_offsets_norm, bba_offset_errors_norm = orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew,
#                                                          n_k1_steps, max_dk1, n_k2_steps, max_dk2, RM=RM2,
#                                                          plot_results=True)
# 
pSC._save_and_check_repr(SC.RING, f'after_orbit_BBA_seed{seed}.repr')

plt.show(block=args.plot)