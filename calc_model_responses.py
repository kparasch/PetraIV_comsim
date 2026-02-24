from pySC import generate_SC

yaml_filepath = 'petra4_conf.yaml'
SC = generate_SC(yaml_filepath, seed=1)
SC.tuning.calculate_model_trajectory_response_matrix(n_turns=1, save_as='model_RM/trajectory1.json')
SC.tuning.calculate_model_trajectory_response_matrix(n_turns=2, save_as='model_RM/trajectory2.json')
SC.tuning.calculate_model_orbit_response_matrix(save_as='model_RM/orbit.json')