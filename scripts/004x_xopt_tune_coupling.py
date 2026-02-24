from xopt.vocs import VOCS
from xopt.evaluator import Evaluator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.stopping_conditions import MaxEvaluationsCondition
from xopt import Xopt
from pySC import SimulatedCommissioning
import numpy as np
import argparse

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    # define variables and function objectives

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_03_seed{seed}.json')


    SC.injection.n_particles = 1

    vocsxy = VOCS(
        variables={SC.tuning.tune.knob_qx: [-2, 2],
                   SC.tuning.tune.knob_qy: [-2, 2],
                   },
        objectives={"f": "MINIMIZE"},
    )
    vocsxyc = VOCS(
        variables={SC.tuning.tune.knob_qx: [-2, 2],
                   SC.tuning.tune.knob_qy: [-2, 2],
                   SC.tuning.c_minus.knob_real: [-0.5, 0.5],
                   SC.tuning.c_minus.knob_imag: [-0.5, 0.5],
                   },
        objectives={"f": "MINIMIZE"},
    )

    #starting_point = SC.magnet_settings.get_many(vocsxy.variable_names)

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
    for vocs in vocsxy, vocsxyc:
        starting_point = SC.magnet_settings.get_many(vocs.variable_names)
        generator = ExpectedImprovementGenerator(vocs=vocs, turbo_controller="optimize")
        max_evals_condition = MaxEvaluationsCondition(max_evaluations=60)
        X = Xopt(evaluator=evaluator, generator=generator, vocs=vocs, stopping_condition=max_evals_condition)
        X.evaluate_data(starting_point)
        X.random_evaluate(30)
        X.run()

        idx, val, best_settings = X.vocs.select_best(X.data)
        SC.magnet_settings.set_many(best_settings)

    gain = 0.6
    n_iter = 10
    for ii in range(n_iter):
        logger.info(f"Iteration {ii+1}/{n_iter}:")
        SC.tuning.tune.correct(n_iter=1, gain=gain, measurement_method='cheat_with_integer')
        SC.tuning.c_minus.correct(gain=gain)
        SC.tuning.correct_orbit(parameter=20, gain=gain)


    threads = args.threads if args.threads > 1 else None
    SC.lattice.omp_num_threads = threads

    SC.injection.n_particles = 512
    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")
    np.savez(f"data/Seeds/injection_efficiency_first_{seed}", injef)

    SC.injection.n_particles = 1
    SC.lattice.omp_num_threads = None

    SC.to_json(f'data/Seeds/pySC_petra4_04b_seed{seed}.json')