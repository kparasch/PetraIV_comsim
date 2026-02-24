from pySC import generate_SC
import json

yaml_filepath = 'petra4_conf.yaml'
SC = generate_SC(yaml_filepath, seed=1)

tune_knob_data = SC.tuning.tune.create_tune_knobs()
tune_knob_data.save_as('model_RM/tune_knobs.json')

cm_knob_data = SC.tuning.c_minus.create_c_minus_knobs()
cm_knob_data.save_as('model_RM/c_minus_knobs.json')
