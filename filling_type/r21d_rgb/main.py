import sys
sys.path.insert(0, '../../filling_level/r21d_rgb/')
from time import localtime, strftime
from main import Config, run_kfold, get_cmd_args
# import argparse

if __name__ == "__main__":
    # Reproduce the best experiment
    # if True, will use the pre-trained model and make predictions, if False, will train the model
    use_pretrained = True
    exp_name = 200903213714
    cfg = Config()
    cfg.load_from(path=f'./predictions/{exp_name}/cfg.txt')
    # replacing the time with the old_time + current_time such that there is no collision
    if use_pretrained:
        cfg.init_time = exp_name
    else:
        cfg.init_time = f'{cfg.init_time}_{strftime("%y%m%d%H%M%S", localtime())}'
    # Expected average of Best Metrics on Each Valid Set: 0.673348 @ 200903213714
    run_kfold(cfg, use_pretrained, get_cmd_args().predict_on_private)

    # Experiment with other parameters
    # cfg = Config()
    # cfg.assign_variable('task', 'ftype')
    # cfg.assign_variable('output_dim', 4)
    # cfg.assign_variable('model_type', 'GRU')
    # cfg.assign_variable('bi_dir', False)
    # cfg.assign_variable('device', 'cuda:1')
    # cfg.assign_variable('data_root', '../../filling_level/r21d_rgb/r21d_rgb_features')
    # cfg.assign_variable('drop_p', 0.0) # results will be irreproducible
    # cfg.assign_variable('batch_size', 64)
    # cfg.assign_variable('input_dim', 512)
    # cfg.assign_variable('hidden_dim', 256)
    # cfg.assign_variable('n_layers', 2)
    # cfg.assign_variable('num_epochs', 30)
    # cfg.assign_variable('seed', 1337)
    # run_kfold(cfg)
