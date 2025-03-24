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
# repr_file = f'after_reramp_seed{seed}.repr'
repr_file = f'after_2nd_tune_orbit_seed{seed}.repr'
SC.RING = pSC._load_repr(repr_file)
SC.RING = pSC.fix_apertures(SC.RING)

SC.INJ.Z0 = np.zeros(6)
SC.INJ.trackMode = 'ORB'

# RM1, RM2 = pSC.load_response_matrices([f'ORM1_{repr_file}.npz', f'ORM2_{repr_file}.npz'])
ORM = np.load(f'ORM_ideal.npz')['arr_0']

mag_ords = np.tile(knobs.bba_sextupoles, (2, 1))
bpm_ords = np.tile(knobs.bpms_disp_bump, (2, 1))
quad_is_skew = True

n_k1_steps = 5
max_dk1 = 10e-6
n_k2_steps = 2
max_dk2 = 0.5e-2

SC.orbits = []
SC.bps = []
SC, bba_offsets_skew, bba_offset_errors_skew = orbit_bba(SC, bpm_ords[:,:4], mag_ords[:,:4], quad_is_skew,
                                                         n_k1_steps, max_dk1, n_k2_steps, max_dk2, RM=ORM,
                                                         plot_results=True)

# mag_ords = np.tile(knobs.bba_quads, (2, 1))
# bpm_ords = np.tile(knobs.bpms_no_disp, (2, 1))
# quad_is_skew = False
# 
# SC, bba_offsets_norm, bba_offset_errors_norm = orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew,
#                                                          n_k1_steps, max_dk1, n_k2_steps, max_dk2, RM=ORM,
#                                                          plot_results=True)

# pSC._save_and_check_repr(SC.RING, f'after_2nd_orbit_BBA_seed{seed}.repr')

slopes = []
intercepts = []
orbits = SC.orbits[2]
bps = SC.bps[2]
cols = ['r','b']
plt.figure()
for k2_step in range(n_k2_steps):
    for bpm in range(orbits.shape[2]):
        orb = orbits[:, k2_step, bpm]
        bp = bps[:, k2_step]
        mask = np.logical_and(~np.isnan(bp), ~np.isnan(orb))
        p = np.polyfit(bp, orb, 1)
        slopes.append(p[0])
        intercepts.append(p[1])
        
for k2_step in range(n_k2_steps):
    for bpi in range(orbits.shape[2]):
        plt.plot(bps[:,k2_step]*1e6, orbits[:, k2_step, bpi]*1e6, '.-', lw=0.5, c=cols[k2_step], alpha=0.3 )
# plt.show()

plt.show()