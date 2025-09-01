import at
from at import Lattice
import numpy as np
from pySC.utils import at_wrapper, logging_tools, sc_tools

import scipy
import matplotlib.pyplot as plt
from pySC.core.constants import SPEED_OF_LIGHT
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.beam import beam_transmission, bpm_reading
from pySC.lattice_properties.response_model import SCgetModelRM
from pySC.utils.sc_tools import ords_from_regex, pinv, randnc
from pySC.correction.orbit_trajectory import (first_turn, stitch, balance, correct)
from pySC.correction.bba import trajectory_bba, _get_bpm_offset_from_mag, plot_bpm_offsets_from_magnets, is_bba_errored, fake_bba
from pySC.plotting.plot_support import plot_support
from pySC.plotting.plot_phase_space import plot_phase_space
from pySC.correction.rf import correct_rf_phase,correct_rf_frequency,phase_and_energy_error
from pySC.correction.tune import tune_scan
from pySC.utils.at_wrapper import findspos
from pySC.plotting.plot_lattice import plot_lattice
from pySC.core.lattice_setting import switch_rf
from pySC.core.lattice_setting import switch_rf

from pySC.utils import sc_tools

LOGGER = logging_tools.get_logger(__name__)

OUT_DIR = ""


class PetraErrorModel:
    # switches
    injection_off_axis_factor = 0
    inj_error_factor = 1
    inj_jitter_factor = 1
    magnet_error_factor = 1
    diag_error_factor = 1
    rf_error_factor = 0
    circ_error_factor = 0

    # bases
    roll_base = np.array([1, 0, 0])
    offset_base = np.array([1, 1, 0])
    offset_base2 = np.array([[1, 1, 0], [1, 1, 0]])
    two_ones = np.ones(2)

    # Magnet errors
    quad_offset = 1 * 30E-6 * offset_base * magnet_error_factor
    quad_roll = 1 * 200E-6 * roll_base * magnet_error_factor
    quad_calibration = 1 * 0.5E-3 * magnet_error_factor

    magnet_offset = 1 * 30E-6 * offset_base * magnet_error_factor
    magnet_roll = 1 * 200E-6 * roll_base * magnet_error_factor
    magnet_calibration = 1 * 1E-3 * magnet_error_factor

    dip_offset = 1 * 30E-6 * offset_base * magnet_error_factor
    dip_roll = 1 * 200E-6 * roll_base * magnet_error_factor
    dip_calibration = 1 * 1E-3 * magnet_error_factor

    section_offset = 0 * 100E-6 * offset_base2 * magnet_error_factor
    girder_offset = 1 * 150E-6 * offset_base2 * magnet_error_factor
    girder_roll = 1 * 200E-6 * roll_base * magnet_error_factor

    #  Errors of diagnostic devices
    bpm_calibration = 1 * 5E-2 * two_ones * diag_error_factor
    bpm_offset = 1 * 500E-6 * two_ones * diag_error_factor
    bpm_noise = 1 * 50E-6 * two_ones * diag_error_factor
    bpm_noise_co = 1 * 1E-7 * two_ones * diag_error_factor
    bpm_roll = 1 * 400E-6 * diag_error_factor
    cm_calibration = 1 * 2E-2 * diag_error_factor
    cm_roll = 1 * 200E-6 * roll_base * diag_error_factor
    cm_limit = 0.5E-3

    # RF cavity errors
    rf_frequency = 1 * 1E2 * rf_error_factor
    rf_voltage = 1 * 1E-3 * 6E6 * rf_error_factor
    rf_time_lag = 1 * 1 / 4 * 0.6 * rf_error_factor

    # Circumference
    circumference = circ_error_factor * 1e-6  # relative

    # Define initial bunch emittance and long. beam size
    # TODO these are optics functions of stored beam, not injected
    emit0 = 20E-9 * two_ones
    betx0, bety0 = 46.02188663, 7.037428502
    alfx0, alfy0 = -0.02174027975, -0.145332546

    beam_size5 = 2E-3
    beam_size6 = 75E-12 * SPEED_OF_LIGHT
    sigx = emit0[0] * np.array([[betx0, -alfx0], [-alfx0, (1 + alfx0 ** 2) / betx0]])
    sigy = emit0[1] * np.array([[bety0, -alfy0], [-alfy0, (1 + alfy0 ** 2) / bety0]])
    sigz = np.diag(np.square(np.array([beam_size5, beam_size6])))
    injection_position = np.zeros(6)
    injection_position[0] = 6e-3
    # Number of particles used for calculating beam transmission at various steps in the chain
    n_part_beam_capture = 50
    # Number of maximum turns used for calculating beam transmission at various steps in the chain
    n_turns_beam_capture = 100
    # Number of angular steps to calculate the dynamic aperture
    dyn_aper_steps = 25

    n_correctors = (762, 762)  # Horizontal and vertical
    n_bpms = 790
    n_quads = 1348


class PetraKnobs:
    def __init__(self, ring: Lattice):
        self.ring: Lattice = ring.deepcopy()
        self.qf = np.array([ind for ind in sc_tools.ords_from_regex(ring, '^QF[0-4]') if (
                ("S" not in ring[ind].FamName) and ("N" not in ring[ind].FamName) and (
                 "O" not in ring[ind].FamName) and ("W" not in ring[ind].FamName)) or ("DW" in ring[ind].FamName)])
        self.qd = np.array([ind for ind in sc_tools.ords_from_regex(ring, '^QD[0-4]') if (
                ("S" not in ring[ind].FamName) and ("N" not in ring[ind].FamName) and (
                 "O" not in ring[ind].FamName) and ("W" not in ring[ind].FamName)) or ("DW" in ring[ind].FamName)])
        self.bpms = np.setdiff1d(sc_tools.ords_from_regex(ring, "BPM"), sc_tools.ords_from_regex(ring, 'BPM_TFB|BPM_LFB|BPM_CUR'))
        self.bpms_disp_bump = sc_tools.ords_from_regex(ring, "^BPM_0[3467]$")
        self.bba_sextupoles = sc_tools.ords_from_regex(ring, "^SD1[ABCD]$")
        self.bpms_no_disp = [ind for ind in self.bpms if ind not in self.bpms_disp_bump]
        self.quads = sc_tools.ords_from_regex(ring, '^QD|^QF')
        s_pos = at.get_s_pos(ring)
        self.bba_quads = self.quads[np.argmin(np.abs(s_pos[self.bpms_no_disp][np.newaxis, :] - s_pos[self.quads][:, np.newaxis]), axis=0)]
        self.bba_magnets = np.sort(np.concatenate((self.bba_sextupoles, self.bba_quads)))


def fix_apertures(ring):
    apertures = [ind for ind in range(len(ring)) if not hasattr(ring[ind], "EApertures")]
    # For simplicity defining aperture of open collimators as EAperture
    for ap in apertures:
        if "COLLX" in ring[ap].FamName:
            ring[ap].EApertures = np.array([0.01, 0.01])
        elif "COLLY" in ring[ap].FamName:
            ring[ap].EApertures = np.array([0.01925, 0.01])
        else:
            raise ValueError("This should not happen, all apertures are defined as E, except for undefined collimators.")
    # TODO 4 mm is correct
    apertures = [ind for ind in range(len(ring)) if np.min(ring[ind].EApertures) == 0.004]
    for ap in apertures:
        if np.allclose(ring[ap].EApertures, np.array([0.01, 0.004])):
            ring[ap].EApertures = np.array([0.01, 0.0035])  # 4mm is actually the outer edge of beam pipe
        else:
            raise ValueError("New element with 4 mm vertical halfgap and not 10mm horizontal?")

    LOGGER.info("Opening in-vacuum devices to 10 mm half gaps.")
    apertures = [ind for ind in range(len(ring)) if np.min(ring[ind].EApertures) == 0.0025]
    for ap in apertures:
        if np.allclose(ring[ap].EApertures, np.array([0.01, 0.0025])):
            ring[ap].EApertures = np.array([0.01, 0.01])  # Opening in-vacuum devices
        else:
            print(ring[ap].EApertures, ring[ap].FamName, ap)
            raise ValueError("New element with 2.5 mm vertical halfgap and not 10mm horizontal?")
    return ring


def register_petra_stuff(file_name, Pem: PetraErrorModel):
    ring = at.load_lattice(file_name)
    ring = fix_apertures(ring)
    ring.enable_6d()
    at.save_repr(ring, f"{file_name[:-4]}.repr")

    SC = SimulatedCommissioning(ring)
    SC.INJ.Z0ideal = Pem.injection_off_axis_factor * Pem.injection_position
    SC.INJ.Z0 = Pem.injection_off_axis_factor * Pem.injection_position
    SC.SIG.randomInjectionZ = Pem.inj_jitter_factor * np.array(
        [20e-6, 20e-6, 1e-6, 1e-6, 1e-4, 0.1 / 360 * SPEED_OF_LIGHT / 5E9])
    SC.SIG.staticInjectionZ = Pem.inj_error_factor * np.array([200e-6, 100e-6, 200e-6, 100e-6, 1e-3, 0])
    SC.INJ.beamSize = scipy.linalg.block_diag(Pem.sigx, Pem.sigy, Pem.sigz)
    SC.SIG.Circumference = Pem.circumference
    SC.INJ.beamLostAt = 0.6  # relative

    # Register pure quadrupoles
    quad_ords = sc_tools.ords_from_regex(SC.RING, '^QD|^QF')
    SC.register_magnets(quad_ords,
                        CalErrorB=np.array([0, Pem.quad_calibration]),
                        MagnetOffset=Pem.quad_offset,
                        MagnetRoll=Pem.quad_roll)
    # Register DQL longitudinal gradient dipole main slice
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^DLQ1_1'), # TODO 60 um
                        CF=1,
                        MagnetOffset=Pem.dip_offset,
                        MagnetRoll=Pem.dip_roll)
    # FAKE register DLQ longitudinal gradient dipole connected slices
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^DLQ1_[2-4]'),
                        CF=1)
    # Register DLGs: split 6-dipoles

    masters = sc_tools.ords_from_regex(SC.RING, '^DQ2_1')
    SC.register_magnets(masters,
                        CF=1,
                        MasterOf=sc_tools.ords_from_regex(SC.RING, '^DQ2_[2-4]').reshape((3, len(masters))),
                        BendingAngle=Pem.dip_calibration,
                        MagnetOffset=Pem.dip_offset,
                        MagnetRoll=Pem.dip_roll)
    masters = sc_tools.ords_from_regex(SC.RING, '^DQ3_1')
    SC.register_magnets(masters,
                        CF=1,
                        MasterOf=sc_tools.ords_from_regex(SC.RING, '^DQ3_[2-6]').reshape((5, len(masters))),
                        BendingAngle=Pem.dip_calibration,
                        MagnetOffset=Pem.dip_offset,
                        MagnetRoll=Pem.dip_roll)
    # Register sextupoles (with skew correctors)
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^SD'),
                        SkewQuad=1,
                        CalErrorB=np.array([0, 0, Pem.magnet_calibration]),
                        CalErrorA=np.array([0, Pem.magnet_calibration, 0]),
                        MagnetOffset=Pem.magnet_offset,
                        MagnetRoll=Pem.magnet_roll)
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^SF'),
                        CalErrorB=np.array([0, 0, Pem.magnet_calibration]),
                        MagnetOffset=Pem.magnet_offset,
                        MagnetRoll=Pem.magnet_roll)
    # Register octupoles
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^OD1|^OF2'),
                        CalErrorB=np.array([0, 0, 0, Pem.magnet_calibration]),
                        MagnetOffset=Pem.magnet_offset,
                        MagnetRoll=Pem.magnet_roll)
    # Register all CMs
    SC.register_magnets(sc_tools.ords_from_regex(SC.RING, '^CX|^CY'),
                        HCM=Pem.cm_limit,
                        VCM=Pem.cm_limit,
                        CalErrorB=np.array([Pem.cm_calibration, 0]),
                        CalErrorA=np.array([Pem.cm_calibration, 0]),
                        MagnetRoll=Pem.magnet_roll)
    # Define slow and fast CMs (these fields are just for convience stored in the SC structure but are not used)
    # What is the name of fast correctors in straights - missing below as they miss F in the name
    SC.ORD.FCM = [sc_tools.ords_from_regex(SC.RING, '^CXYSF|^CXSF'), sc_tools.ords_from_regex(SC.RING, '^CXYSF|^CYSF')]
    SC.ORD.SCM = [sc_tools.ords_from_regex(SC.RING, '^CXYS|^CXS|^CYS'), sc_tools.ords_from_regex(SC.RING, '^CXYS|^CXS|^CYS')]

    # Set the slow CMs to be used by the toolkit
    SC.ORD.HCM = SC.ORD.SCM[0].copy()
    SC.ORD.VCM = SC.ORD.SCM[1].copy()

    # Register BPMs
    SC.register_bpms(np.setdiff1d(sc_tools.ords_from_regex(SC.RING, "BPM"), sc_tools.ords_from_regex(SC.RING, 'BPM_TFB|BPM_LFB|BPM_CUR')),
                     CalError=Pem.bpm_calibration,
                     Offset=Pem.bpm_offset,
                     Noise=Pem.bpm_noise,
                     NoiseCO=Pem.bpm_noise_co,
                     Roll=Pem.bpm_roll)
    # Register Cavities
    SC.register_cavities(np.argwhere(at.get_cells(SC.RING, "Frequency"))[0],
                         FrequencyOffset=Pem.rf_frequency,
                         VoltageOffset=Pem.rf_voltage,
                         TimeLagOffset=Pem.rf_time_lag)
    # Register Girgers
    SC.register_supports(np.vstack((sc_tools.ords_from_regex(SC.RING, 'GS'), sc_tools.ords_from_regex(SC.RING, 'GE'))),
                         "Girder",
                         Offset=Pem.girder_offset,
                         Roll=Pem.girder_roll)

    # Copy registered elements to IDEALRING
    # SC.IDEALRING = SC.RING.deepcopy()

    assert SC.ORD.HCM.shape[0] == Pem.n_correctors[0]
    assert SC.ORD.VCM.shape[0] == Pem.n_correctors[1]
    assert SC.ORD.BPM.shape[0] == Pem.n_bpms
    assert quad_ords.shape[0] == Pem.n_quads

    return SC


def number_of_elements(SC):
    LOGGER.info(f"Number of correctors: {len(SC.ORD.HCM)}, {len(SC.ORD.VCM)}")
    LOGGER.info(f"Number of BPMs: {len(SC.ORD.BPM)}")


def apply_errors_and_init(SC, plot=False):
    _save_and_check_repr(SC.RING, "repr_bare.repr")
    if plot:
        plot_lattice(SC, s_range=(510, 525), plot_magnet_names=True)
    SC.apply_errors()
    _save_and_check_repr(SC.RING, "repr_initial_errors.repr")
    if plot:
        plot_support(SC)

    # Switch off sextupoles and octupoles and cavities
    SC.RING = switch_rf(SC.RING, SC.ORD.RF, False)
    SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^SF|^SD'), 0.0, False, 2, method='abs')
    #SC.set_magnet_setpoints(SCgetOrds(SC.RING, '^OD|^OF'), 0.0, False, 3, method='abs')
    return SC


def _save_and_check_repr(ring, file_name):
    at.save_repr(ring, f"{OUT_DIR}{file_name}")
    ring2 = at.load_lattice(f"{OUT_DIR}{file_name}")
    ring2.enable_6d()
    test = SimulatedCommissioning(ring2)


def _load_repr(file_name):
    ring = at.load_lattice(file_name)
    ring.enable_6d()
    return ring


def calculate_and_save_response_matrices(SC, file_names):
    rm2 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=2, useIdealRing=True)
    nbpm4, ncor = rm2.shape
    rm1 = rm2.reshape(2, 2, nbpm4 // 4, ncor)[:, 0, :, :].reshape(nbpm4 // 2, ncor)
    np.savez(f"{OUT_DIR}{file_names[0]}", rm1)
    np.savez(f"{OUT_DIR}{file_names[1]}", rm2)
    return rm1, rm2


def load_response_matrices(file_names):
    return (np.load(f"{OUT_DIR}{file_names[0]}")["arr_0"],
            np.load(f"{OUT_DIR}{file_names[1]}")["arr_0"])


def beam_threading(SC, Pem, run_rm):
    SC.RING = switch_rf(SC.RING, SC.ORD.RF, False)
    # Switch off sextupoles and octupoles
    SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^SF|^SD'), 0.0, False, 2, method='abs')
    SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^OD|^OF'), 0.0, False, 3, method='abs')
    SC.INJ.trackMode = "TBT"
    #SC.plot = True
    trajectory_rm_names = ("rm1.npz", "rm2.npz")
    RM1, RM2 = (calculate_and_save_response_matrices(SC, trajectory_rm_names)
                if run_rm else load_response_matrices(trajectory_rm_names))
    SC.INJ.nParticles = 1
    SC.INJ.nTurns = 1
    SC.INJ.nShots = 1
    SC.INJ.trackMode = 'TBT'
    eps = 1E-4  # Noise level

    SC = first_turn(SC, RM1, alpha=100, maxsteps=200)
    max_turns, fraction_lost = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                 nTurns=Pem.n_turns_beam_capture, plot=True)
    for alpha in (100, 50, 20, 5):
        LOGGER.info(f"{alpha=}")
        SC = correct(SC, RM1, alpha=alpha, target=50e-6, maxsteps=200, eps=eps)
        max_turns, fraction_lost = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                     nTurns=Pem.n_turns_beam_capture, plot=True)

    SC = correct(SC, RM1, alpha=5, target=50e-6, maxsteps=200, eps=eps)

    SC.INJ.nTurns = 2
    for n_bpms in (20, 10, 5, 3):
        try:
            SC = stitch(SC, RM2, alpha=20, n_bpms=n_bpms, maxsteps=200)
        except RuntimeError:
            pass
        else:
            break
    else:
        raise RuntimeError("No stitching attempt worked.")

    SC = correct(SC, RM2, alpha=20, target=50e-6, maxsteps=50, eps=eps)
    SC = balance(SC, RM2, alpha=20, maxsteps=32, eps=eps)
    max_turns, fraction_lost = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                 nTurns=Pem.n_turns_beam_capture, plot=True)
    return SC


def perform_trajectory_bba(SC, knobs, n_bpms=None, only_failed=False):
    skew_ords = np.tile(knobs.bba_sextupoles, (2, 1))
    bpms_skew = np.tile(knobs.bpms_disp_bump, (2, 1))
    failed_bpms = []
    failed_quads = []
    if n_bpms is not None:
        skew_ords = skew_ords[:, :n_bpms]
        bpms_skew = bpms_skew[:, :n_bpms]
    if only_failed:
        assert n_bpms is None
        for ii, bpmid in enumerate(bpms_skew[0]):
            has_BBA = SC.RING[bpmid].has_BBA 
            if not has_BBA[0] or not has_BBA[1]:
                failed_bpms.append(bpmid)
                failed_quads.append(skew_ords[0][ii])
        
        skew_ords = np.tile(np.array(failed_quads), (2,1))
        bpms_skew = np.tile(np.array(failed_bpms), (2,1))
        print(f'Retrying {len(bpms_skew[0])} BPMs.')

    SC, bba_offsets, bba_offset_errors = trajectory_bba(
        SC, bpms_skew, skew_ords, n_steps=6, num_downstream_bpms=50,
        q_ord_phase=knobs.quads[2], q_ord_setpoints=np.array([0.95, 0.8, 0.7, 1.0, 1.05]),
        max_injection_pos_angle=np.array([1.5E-3, 1.5E-3]),
        magnet_order=1, fit_order=1, magnet_strengths=np.array([-0.1, 0.1]), setpoint_method='add',
        plot_results=True, skewness=True, dipole_compensation=False)
    final_offsets = _get_bpm_offset_from_mag(SC.RING, bpms_skew, skew_ords)
    plot_bpm_offsets_from_magnets(SC, bpms_skew, skew_ords, is_bba_errored(bba_offsets, bba_offset_errors))
    print("SKEW", np.std(final_offsets, axis=1))
    np.savez(f"{OUT_DIR}bba_skew", OFF=bba_offsets, ERR=bba_offset_errors)

    normal_ords = np.tile(knobs.bba_quads, (2, 1))
    bpms_normal = np.tile(knobs.bpms_no_disp, (2, 1))
    if n_bpms is not None:
        normal_ords = normal_ords[:, :n_bpms]
        bpms_normal = bpms_normal[:, :n_bpms]
    if only_failed:
        assert n_bpms is None
        for ii, bpmid in enumerate(bpms_normal[0]):
            has_BBA = SC.RING[bpmid].has_BBA 
            if not has_BBA[0] or not has_BBA[1]:
                failed_bpms.append(bpmid)
                failed_quads.append(normal_ords[0][ii])
        
        normal_ords = np.tile(np.array(failed_quads), (2,1))
        bpms_normal = np.tile(np.array(failed_bpms), (2,1))
        print(f'Retrying {len(bpms_skew[0])} BPMs.')

    SC, bba_offsets, bba_offset_errors = trajectory_bba(
        SC, bpms_normal, normal_ords, n_steps=6, num_downstream_bpms=50,
        q_ord_phase=knobs.quads[2], q_ord_setpoints=np.array([0.95, 0.8, 0.7, 1.0, 1.05]),
        magnet_order=1, fit_order=1, magnet_strengths=np.array([-0.2, 0.2]), setpoint_method='add',
        plot_results=True, skewness=False, dipole_compensation=False, max_injection_pos_angle=np.array([1.5E-3, 1.5E-3]))
    final_offsets = _get_bpm_offset_from_mag(SC.RING, bpms_normal, normal_ords)
    plot_bpm_offsets_from_magnets(SC, bpms_normal, normal_ords, is_bba_errored(bba_offsets, bba_offset_errors))
    print("NORMAL", np.std(final_offsets, axis=1))
    final_offsets = _get_bpm_offset_from_mag(SC.RING, np.tile(knobs.bpms, (2, 1)), np.tile(knobs.bba_magnets, (2, 1)))
    np.savez(f"{OUT_DIR}bba_normal", OFF=bba_offsets, ERR=bba_offset_errors)
    print(np.std(final_offsets, axis=1))
    return SC


def remove_bba_outliers(SC, knobs, limit=1e-4):
    bpms_2d = np.tile(knobs.bpms, (2, 1))
    magnets_2d = np.tile(knobs.bba_magnets, (2, 1))
    final_offsets = _get_bpm_offset_from_mag(SC.RING, bpms_2d, magnets_2d)
    errors = np.zeros(final_offsets.shape)
    errors[np.abs(final_offsets) > limit] = 1
    #SC = _fake_measurement(SC, bpms_2d, magnets_2d, errors)
    final_offsets = _get_bpm_offset_from_mag(SC.RING, bpms_2d, magnets_2d)
    print(np.std(final_offsets, axis=1))
    return SC


def assumed_bba_result(SC, Pem, bba_offset):
    for nBPM in SC.ORD.BPM:
        SC.RING[nBPM].Offset = bba_offset * Pem.two_ones * randnc(shape=bba_offset.shape)
    return SC


def multipole_ramp_up(SC, Pem, run_rm, nsteps, alpha=50):
    trajectory_rm_names = ("rm1.npz", "rm2.npz")
    RM1, RM2 = (calculate_and_save_response_matrices(SC, trajectory_rm_names)
                if run_rm else load_response_matrices(trajectory_rm_names))
    SC.INJ.nTurns = 2
    eps = 1E-4  # Noise level
    SC = correct(SC, RM2, alpha=alpha, target=50e-6, maxsteps=50, eps=eps)
    SC = balance(SC, RM2, alpha=alpha, damping=0.3, maxsteps=32, eps=eps)
    max_turns, fraction_survived = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                     nTurns=Pem.n_turns_beam_capture,
                                                     plot=True)
    for setp in np.arange(nsteps + 1) / nsteps:
        LOGGER.info(f"Sextupoles and octupoles at {setp} of nominal")
        SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^SF|^SD'), setp, False, 2,  method='rel')
        # TODO at which point are octupoles actually needed
        SC.set_magnet_setpoints(sc_tools.ords_from_regex(SC.RING, '^OD|^OF'), setp, False, 3,  method='rel')
        SC = balance(SC, RM2, alpha=alpha, maxsteps=32, eps=eps)
        max_turns, fraction_survived = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                     nTurns=Pem.n_turns_beam_capture,
                                                     plot=True)

    SC = balance(SC, RM2, alpha=alpha, maxsteps=32, eps=eps)

    max_turns, fraction_survived = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                 nTurns=Pem.n_turns_beam_capture,
                                                 plot=True)
    return SC


def compare_setpoints_ideal(SC):
    for ord in SC.ORD.Magnet:
        for letter in ("A", "B"):
            if not np.allclose(getattr(SC.RING[ord], f"SetPoint{letter}"), getattr(SC.IDEALRING[ord], f"Polynom{letter}")):
                LOGGER.warning(f"{ord} {SC.RING[ord].FamName} {letter}")


def setup_rf(SC, Pem, knobs):
    plot_phase_space(SC, nParticles=10, nTurns=100)
    SC.RING = switch_rf(SC.RING, SC.ORD.RF, True)
    # Plot initial phasespace
    plot_phase_space(SC, nParticles=10, nTurns=100)
    # RF cavity correction
    phase_and_energy_error(SC, SC.ORD.RF)
    LOGGER.info("Start of RF optimisation")
    for nIter in range(3):
        SC.INJ.nTurns = 5
        SC = correct_rf_phase(SC, bpm_ords=knobs.bpms_disp_bump, n_steps=25, plot_results=True, plot_progress=False)
        SC.INJ.nTurns = 15
        SC = correct_rf_frequency(SC, bpm_ords=knobs.bpms_disp_bump, n_steps=15, f_range=500 * np.array([-1, 1]),
                                  plot_results=True, plot_progress=False)
        phase_and_energy_error(SC, SC.ORD.RF)
    _save_and_check_repr(SC.RING, f"after_rf.repr")
    # # Plot phasespace after RF correction
    plot_phase_space(SC, nParticles=10, nTurns=100)
    max_turns, fraction_lost = beam_transmission(SC, nParticles=Pem.n_part_beam_capture,
                                                 nTurns=Pem.n_turns_beam_capture,
                                                 plot=True)
    return SC


def tune_scan_transmission(SC, Pem, knobs):
    SC, tsetpoints, tmax_turns, transmission = tune_scan(
        SC, [knobs.qf, knobs.qd],
        np.outer(np.ones(2), 1 + np.linspace(-0.001, 0.001, 51)), do_plot=True, nParticles=Pem.n_part_beam_capture,
        nTurns=Pem.n_turns_beam_capture, target=0.97)
    LOGGER.info(tsetpoints)
    _save_and_check_repr(SC.RING, f"after_tune_scan.repr")
    np.savez(f"{OUT_DIR}tune_scan_transmission.npy", MAX_TURNS=tmax_turns, TRANS=transmission)
    return SC



def run_beam_threading(SC, Pem, run_rm, run_threading):
    if run_threading:
        SC = beam_threading(SC, Pem, run_rm)
        _save_and_check_repr(SC.RING, "after_threading.repr")
    else:
        SC.RING = _load_repr(f"{OUT_DIR}after_threading.repr")
        SC.RING = fix_apertures(SC.RING)
    return SC


def run_multipoles(SC, Pem, run_rm, ramp_multipoles, nsteps, alpha=50):
    if ramp_multipoles:
        SC = multipole_ramp_up(SC, Pem, run_rm, nsteps, alpha=alpha)
        _save_and_check_repr(SC.RING, f"after_multipoles.repr")
    else:
        SC.RING = _load_repr(f"{OUT_DIR}after_multipoles.repr")
        SC.RING = fix_apertures(SC.RING)
    return SC

