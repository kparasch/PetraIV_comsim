import petra_SC as pSC
import nafflib
import at
from pySC.utils import logging_tools, sc_tools
import numpy as np
from pySC.core.beam import bpm_reading
from pySC.correction.orbit_trajectory import correct
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
p424 = 'p4_H6BA_v4_2_4.mat'
SC = pSC.register_petra_stuff(p424, Pem)
pSC.number_of_elements(SC)
knobs = pSC.PetraKnobs(SC.RING)
SC.plot = False
repr_file = f'after_bpm_02_08_BBA_seed{seed}.repr'
SC.RING = pSC._load_repr(repr_file)
SC.RING = pSC.fix_apertures(SC.RING)
SC.INJ.Z0 = np.zeros(6)

# RM1, RM2 = pSC.load_response_matrices([f'ORM1_seed{seed}.npz', f'ORM2_seed{seed}.npz'])
ORM = np.load(f'ORM_ideal.npz')['arr_0']
#RM1, RM2 = pSC.load_response_matrices([f'ORM1_{repr_file}.npz', f'ORM2_{repr_file}.npz'])

SC.INJ.trackMode = 'ORB'
bp0, tr = bpm_reading(SC)
LOGGER.info(f"R.m.s. orbit: x: {np.std(bp0[0])*1e6:.1f} um, y: {np.std(bp0[1])*1e6:.1f} um")


ring = SC.IDEALRING
qf1 = np.array([ind for ind in sc_tools.ords_from_regex(ring, '^QF[1]') if (
    ("S" not in ring[ind].FamName) and ("N" not in ring[ind].FamName) and (
     "O" not in ring[ind].FamName) and ("W" not in ring[ind].FamName)) or ("DW" in ring[ind].FamName)])
qd02 = np.array([ind for ind in sc_tools.ords_from_regex(ring, '^QD[0-2]') if (
    ("S" not in ring[ind].FamName) and ("N" not in ring[ind].FamName) and (
     "O" not in ring[ind].FamName) and ("W" not in ring[ind].FamName)) or ("DW" in ring[ind].FamName)])

def tune_response(SC, q_knob, dk = 1e-5):
    all_refs = np.ones(len(SC.IDEALRING), dtype=bool)
    q1_i, q2_i, _ = SC.IDEALRING.get_optics()[1]['tune']

    for ind in q_knob:
        SC.IDEALRING[ind].K += dk

    q1_f, q2_f, _ = SC.IDEALRING.get_optics()[1]['tune']

    for ind in q_knob:
        SC.IDEALRING[ind].K -= dk
    
    dq1 = (q1_f - q1_i)/dk
    dq2 = (q2_f - q2_i)/dk
    return dq1, dq2



def get_tbt_data(SC, kick=1e-6):
    tbt_data = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns), np.nan)
    tbt_data = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns), np.nan)

    SC.INJ.Z0[1] = kick
    tbt_data_x, transm = bpm_reading(SC)
    SC.INJ.Z0[1] = 0
    SC.INJ.Z0[3] = kick
    tbt_data_y, transm = bpm_reading(SC)
    SC.INJ.Z0[3] = 0

    for turn in range(SC.INJ.nTurns):
        tbt_data[0, :, turn] = tbt_data_x[0, turn*len(SC.ORD.BPM):(turn+1)*len(SC.ORD.BPM)]
        tbt_data[1, :, turn] = tbt_data_y[1, turn*len(SC.ORD.BPM):(turn+1)*len(SC.ORD.BPM)]

    return tbt_data


def measure_tune(SC, kick=1e-6):
    trackMode = SC.INJ.trackMode
    SC.INJ.trackMode = 'TBT'
    SC.INJ.nTurns = 100 
    tbt_data = get_tbt_data(SC, kick=kick)
    
    NN = SC.INJ.nTurns
    while np.sum(np.isnan(tbt_data[0,0,:NN])) or np.sum(np.isnan(tbt_data[1,0,:NN])):
       LOGGER.info(f'Lost in less than {NN} turns, reducing tracking turns to {NN//2}') 
       NN = NN//2
    tbt_data = tbt_data[:,:,:NN]

    tunes = np.full((2, len(SC.ORD.BPM)), np.nan)
    for ii in range(2):
        for jj in range(len(SC.ORD.BPM)):
            tbt_data[ii, jj, :] = tbt_data[ii, jj, :] - np.mean(tbt_data[ii, jj, :])
            tunes[ii, jj] = abs(nafflib.tune(tbt_data[ii, jj, :], window_order=0))
    
    SC.INJ.trackMode = trackMode
    tune = np.nanmean(tunes, axis=1)
    return tune

def correct_tune(SC, target_tune, iTRM, qf, qd, gain=0.8, niter=1):
    for ii in range(niter):
        tune = measure_tune(SC)
        tune_error = tune - target_tune
        LOGGER.info(f'Iteration {ii}, measured tune:  {tune[0]:.4f}, {tune[1]:.4f}')
        knob_error = np.dot(iTRM, -tune_error)
        for ind in qf:
            k1 = SC.RING[ind].SetPointB[1]
            SC.set_magnet_setpoints(ind, k1 + knob_error[0]*gain, skewness=False, order=1, method='abs')
        for ind in qd:
            k1 = SC.RING[ind].SetPointB[1]
            SC.set_magnet_setpoints(ind, k1 + knob_error[1]*gain, skewness=False, order=1, method='abs')
    tune = measure_tune(SC)
    LOGGER.info(f'Final measured tune: {tune[0]:.4f}, {tune[1]:.4f}')
    return 

ideal_tune = np.array([.18, .27])
TRM = np.zeros((2,2))
TRM[:, 0] = tune_response(SC, qf1)
TRM[:, 1] = tune_response(SC, qd02)
iTRM = np.linalg.inv(TRM)



SC.INJ.trackMode = 'ORB'
alpha=25

for _ in range(3):
    bp1, tr = bpm_reading(SC)
    correct_tune(SC, ideal_tune, iTRM, qf1, qd02, gain=0.8, niter=5)
    LOGGER.info(f"R.m.s. orbit: x: {np.std(bp1[0])*1e6:.1f} um, y: {np.std(bp1[1])*1e6:.1f} um")
    correct(SC, ORM, alpha=alpha, maxsteps=100)
    LOGGER.info(f'Orbit correction with alpha = {alpha:.1f}.')
    bp1, tr = bpm_reading(SC)
    LOGGER.info(f"R.m.s. orbit: x: {np.std(bp1[0])*1e6:.1f} um, y: {np.std(bp1[1])*1e6:.1f} um")

fig = plt.figure(0)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(bp0[0], '.')
ax1.plot(bp1[0], '.')
ax2.plot(bp0[1], '.')
ax2.plot(bp1[1], '.')
pSC._save_and_check_repr(SC.RING, f'after_3rd_tune_orbit_seed{seed}.repr')

plt.show(block=args.plot)