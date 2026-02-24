import argparse
from pySC import generate_SC
import matplotlib.pyplot as plt
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument('--factor', type=float, default=1)
argparser.add_argument('--seed', type=int, default=1)
args = argparser.parse_args()

scale_errors = args.factor
seed = args.seed
yaml_filepath = 'petra4_conf.yaml'
SC = generate_SC(yaml_filepath, seed=args.seed, scale_errors=scale_errors, sigma_truncate=2)

SC.import_knob('model_RM/tune_knobs.json')
SC.import_knob('model_RM/c_minus_knobs.json')

SC.to_json(f'data/Seeds/pySC_petra4_00_seed{seed}.json')
