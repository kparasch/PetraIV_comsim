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
repr_file = f'after_2nd_tune_orbit_seed{seed}.repr'
#repr_file = f'after_second_orbit_BBA_seed{seed}.repr'
SC.RING = pSC._load_repr(repr_file)
SC.RING = pSC.fix_apertures(SC.RING)

SC.INJ.Z0 = np.zeros(6)
SC.INJ.trackMode = 'ORB'

# RM1, RM2 = pSC.load_response_matrices([f'ORM1_{repr_file}.npz', f'ORM2_{repr_file}.npz'])
ORM = np.load(f'ORM_ideal.npz')['arr_0']

bpm5_ids = []
bpm5_quads = []
for bpm_id, magnet_id in zip(knobs.bpms_no_disp, knobs.bba_quads):
    if SC.RING[bpm_id].FamName == 'BPM_05':
        bpm5_ids.append(bpm_id)
        bpm5_quads.append(magnet_id)
bpm5_ids = np.tile(bpm5_ids, (2, 1))
bpm5_quads = np.tile(bpm5_quads, (2, 1))
quad_is_skew = False

n_k1_steps = 5
max_dk1 = 20e-6
# max_x = 100e-6
n_k2_steps = 2
max_dk2 = 10e-2


SC.orbits = []
SC.bps = []
SC, bba_offsets_skew, bba_offset_errors_skew = orbit_bba(SC, bpm5_ids, bpm5_quads, quad_is_skew=quad_is_skew,
                                                         n_k1_steps=n_k1_steps, max_dk1=max_dk1,# max_x=max_x, 
                                                         n_k2_steps=n_k2_steps, max_dk2=max_dk2, 
                                                         RM=ORM, plot_results=True)


pSC._save_and_check_repr(SC.RING, f'after_bpm05_BBA_seed{seed}.repr')

# orbits = SC.orbits[1]
# bps = SC.bps[1]
# cols = ['r','b']
# plt.figure()
# for k2_step in range(n_k2_steps):
#     for bpi in range(orbits.shape[2]):
#         plt.plot(bps[:,k2_step]*1e6, orbits[:, k2_step, bpi]*1e6, '.-', c=cols[k2_step] )
# plt.show()

plt.show()