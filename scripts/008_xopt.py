#   def get_chi2(args):
#       #q0, qs, ae, dae, xi, phase_0, amp_scale = args
#       q0 = args["q0"]
#       qs = args["qs"]
#       ae = args["ae"]
#       dae = args["dae"]
#       xi = args["xi"]
#       phase_0 = args["phase_0"]
#       amp_scale = args["amp_scale"]
#       btf = vlasov.get_btf(data_freq, q0, qs, ae, dae, xi)
#       est_amp = np.abs(btf)
#       est_phase = np.angle(btf) + phase_0
#       chi2 = np.sum((est_phase - data_phase)**2)
#       chi2 += amp_weight*np.sum((est_amp - amp_scale * data_amp)**2)
#       chi2 /= 1 + amp_weight
#       print(f"{chi2:=.3f}, {q0:=.4f}, {qs:=.4f}, {ae:=.2e}. {dae:=.2e}, {xi:=.2f}, {phase_0:=.1f}")
#       return chi2

from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.stopping_conditions import MaxEvaluationsCondition
from xopt import Xopt
from pySC import SimulatedCommissioning
import numpy as np


if __name__ == '__main__':
    # define variables and function objectives

    seed = 1
    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_03_seed{seed}.json')
    SC.injection.n_particles = 100#512
    SC.lattice.omp_num_threads = 8

    vocs = VOCS(
        variables={SC.tuning.tune.knob_qx: [-2, 2],
                   SC.tuning.tune.knob_qy: [-2, 2],
                   SC.tuning.c_minus.knob_real: [-0.5, 0.5],
                   SC.tuning.c_minus.knob_imag: [-0.5, 0.5],
                   },
        objectives={"f": "MINIMIZE"},
    )

    starting_point = SC.magnet_settings.get_many(vocs.variable_names)

    def injection_efficiency(input_dict):
        SC.magnet_settings.set_many(input_dict)
        injef = SC.tuning.injection_efficiency(n_turns=500)
        #print(input_dict, injef[-1]*100)
        return {"f": injef[-1]*100}

    def tune_coupling(input_dict):
        SC.magnet_settings.set_many(input_dict)
        try:
            tune_x, tune_y = SC.tuning.tune.cheat_with_integer()
            cm = SC.tuning.c_minus.cheat()
        except:
            tune_x = 0
            tune_y = 0
            cm = 100+100j

        cmr = cm.real
        cmi = cm.imag
        design_qx = SC.tuning.tune.design_qx + SC.tuning.tune.integer_qx
        design_qy = SC.tuning.tune.design_qy + SC.tuning.tune.integer_qy

        chi2 = (design_qx - tune_x)**2 + (design_qy - tune_y)**2 + cmr**2 + cmi**2
        #print(tune_x, tune_y, cmr, cmi, chi2, input_dict)
        return {"f": chi2}

    evaluator = Evaluator(function=tune_coupling)
    generator = ExpectedImprovementGenerator(vocs=vocs, turbo_controller="optimize")
    max_evals_condition = MaxEvaluationsCondition(max_evaluations=200)
    X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs, stopping_condition=max_evals_condition)
    X.evaluate_data(starting_point)
    X.random_evaluate(50)
    X.run()

    idx, val, best_settings = X.vocs.select_best(X.data)
    SC.magnet_settings.set_many(best_settings)

    SC.injection.n_particles = 512
    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    SC.to_json(f'data/Seeds/pySC_petra4_04_seed{seed}.json')
    np.savez(f"data/Seeds/injection_efficiency_before_orbit{seed}", injef)