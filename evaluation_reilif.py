import argparse
import json
import pickle
import random

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from datasets import create_dataset
from models import create_model
import numpy as np
import optuna
"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
"""


def load_data_from_csv_to_dataframe(configuration: dict):
    """Load the data used in the experimental study.
    """

    # load json file with US Regions - US States jurisdiction
    file_hhs_regions_to_states = open(configuration['dataset_path'] + 'HHS_Regions.json', "r")
    dict_hhs_regions_to_states = json.loads(file_hhs_regions_to_states.read())

    print('The dictionary dict_HHSRegions_to_States:', dict_hhs_regions_to_states)

    # load the data used in the experimental study from the file below
    print('Load the data used in the experimental study from the file below')
    regions_used = []
    if configuration['dataset_mode'] == 'us_regions':
        dataset = pd.read_csv(configuration['dataset_path'] + 'regional_timeseries_HHS_Regions.csv')
        print('dataset v0', dataset.shape, '\n', dataset)
        for states in dict_hhs_regions_to_states.keys():
            regions_used.append(states)
    else:
        dataset = pd.read_csv(configuration['dataset_path'] + 'regional_timeseries_US_States.csv')
        print('dataset v0', dataset.shape, '\n', dataset)
        for states in dict_hhs_regions_to_states.values():
            regions_used.extend(states)
        regions_used.append('New York City')

    print('The regions/states in the dataset', regions_used, len(regions_used))

    print("Add the required 'time_idx' column to the dataset - used by TimeSeriesDataSet (in case we use it)")
    # dataset.shape[0] -> gives the total number of weeks through all years (numer of rows)
    dataset["time_idx"] = np.tile(np.arange(dataset.shape[0]), 1)
    # print(dataset)

    print("Add the 'group' column to the dataset - used by TimeSeriesDataSet (in case we use it)")
    # Here, we have only one group for each target variable, i.e., constant value
    dataset["group"] = 0

    # Drop the unnecessary column 'Unnamed: 0'
    dataset.drop('Unnamed: 0', inplace=True, axis=1)
    dataset.to_csv(configuration['dataset_path'] + 'input_data.csv', sep=',')
    print('dataset v1', dataset.shape, '\n', dataset)
    print(dataset.describe())

    # Drop columns 'YEAR', 'WEEK'
    dataset = dataset.drop(['YEAR', 'WEEK'], axis=1)
    print(dataset.describe())

    var_names_ids = {}
    state_data = {}
    state_data_df = {}
    for st_id, st_used in enumerate(regions_used):
        var_id = 0
        state_data[st_id] = []
        var_names_ids[st_id] = {'vars': {}, 'context': {}}

        state_data[st_id].append(dataset[['ILITOTAL_' + st_used]])
        var_names_ids[st_id]['vars']['ILITOTAL'] = var_id
        var_id += 1

        if configuration['variables']['u10']:
            state_data[st_id].append(dataset['u10_' + st_used])
            var_names_ids[st_id]['vars']['u10'] = var_id
            var_id += 1
        if configuration['variables']['v10']:
            state_data[st_id].append(dataset['v10_' + st_used])
            var_names_ids[st_id]['vars']['v10'] = var_id
            var_id += 1
        if configuration['variables']['t2m']:
            state_data[st_id].append(dataset['t2m_' + st_used])
            var_names_ids[st_id]['vars']['t2m'] = var_id
            var_id += 1
        if configuration['variables']['resident_population']:
            state_data[st_id].append(dataset['RESIDENT_POPULATION_' + st_used])
            var_names_ids[st_id]['context']['RESIDENT_POPULATION'] = var_id
            var_id += 1
        if configuration['variables']['resident_population_density']:
            state_data[st_id].append(dataset['RESIDENT_POPULATION_DENSITY_' + st_used])
            var_names_ids[st_id]['context']['RESIDENT_POPULATION_DENSITY'] = var_id
            var_id += 1
        if configuration['variables']['lat']:
            state_data[st_id].append(dataset['LAT_' + st_used])
            var_names_ids[st_id]['context']['LAT'] = var_id
            var_id += 1
        if configuration['variables']['long']:
            state_data[st_id].append(dataset['LONG_' + st_used])
            var_names_ids[st_id]['context']['LONG'] = var_id
            var_id += 1

        print('state_data[st_id]', state_data[st_id])

        state_data_df[st_id] = pd.concat(state_data[st_id], axis=1)

    return state_data_df, regions_used, var_names_ids


def split_and_scale_data(dataset: dict, configuration: dict, regions_used: list, dict_vars_to_ids: dict):
    """Scale dataset given the configuration (loaded from the json file).
    """
    train_data = {}
    val_data = {}
    test_data = {}
    min_max_per_state = {}
    for st_id, st_used in enumerate(regions_used):
        print('Number of time series data instances:', len(dataset[st_id]))
        train_size = int(len(dataset[st_id]) * configuration['train_ratio'])
        print('train_size:', train_size)
        val_size = int(len(dataset[st_id]) * configuration['val_ratio'])
        print('val_size:', val_size)
        test_size = len(dataset[st_id]) - train_size - val_size
        print('test_size:', test_size)

        train_state_data = dataset[st_id].iloc[0:train_size, :].copy()
        val_state_data = dataset[st_id].iloc[train_size:train_size + val_size, :].copy()
        test_state_data = dataset[st_id].iloc[train_size + val_size:len(dataset[st_id]), :].copy()

        print('train_state_data', train_state_data.shape, 'val_state_data', val_state_data.shape, 'test_state_data',
              test_state_data.shape)

        print('train_state_data.columns.tolist()', train_state_data.columns.tolist())
        print('=')
        print("dict_vars_to_ids['st_id']", dict_vars_to_ids[st_id])

        train_data[st_id] = np.zeros_like(train_state_data.to_numpy(), dtype=float)
        val_data[st_id] = np.zeros_like(val_state_data.to_numpy(), dtype=float)
        test_data[st_id] = np.zeros_like(test_state_data.to_numpy(), dtype=float)

        scaler_dict = {}
        for col_id in range(train_state_data.shape[1]):
            # scale each column separately - here the fit for scaling
            if col_id not in dict_vars_to_ids[st_id]['context'].values():
                scmm = MinMaxScaler()
                if configuration['normalize_all_data_from_beginning']:
                    scmm.fit(np.concatenate((np.concatenate((train_state_data.iloc[:, col_id].copy(),
                                                             val_state_data.iloc[:, col_id].copy()), axis=0),
                                             test_state_data.iloc[:, col_id].copy()), axis=0).reshape(-1, 1))
                else:
                    scmm.fit(train_state_data.iloc[:, col_id].copy().to_numpy().reshape(-1, 1))
                train_data[st_id][:, col_id] = scmm.transform(train_state_data.iloc[:, col_id].copy().to_numpy()
                                                              .reshape(-1, 1)).ravel()
                val_data[st_id][:, col_id] = scmm.transform(val_state_data.iloc[:, col_id].copy().to_numpy()
                                                            .reshape(-1, 1)).ravel()
                test_data[st_id][:, col_id] = scmm.transform(test_state_data.iloc[:, col_id].copy().to_numpy()
                                                             .reshape(-1, 1)).ravel()
                scaler_dict[col_id] = scmm
            else:
                train_data[st_id][:, col_id] = train_state_data.iloc[:, col_id].copy().to_numpy()
                val_data[st_id][:, col_id] = val_state_data.iloc[:, col_id].copy().to_numpy()
                test_data[st_id][:, col_id] = test_state_data.iloc[:, col_id].copy().to_numpy()

        print('train_data[st_id]', train_data[st_id].shape)
        print('val_data[st_id]', val_data[st_id].shape)
        print('test_data[st_id]', test_data[st_id].shape)
        print('scaler_dict', scaler_dict)
        min_max_per_state[st_id] = scaler_dict[dict_vars_to_ids[st_id]['vars']['ILITOTAL']]

    return train_data, val_data, test_data, min_max_per_state


def print_current_losses(epoch, max_epochs, iter, max_iters, losses):
    """Print current losses on console.

        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
    message = '[epoch: {}/{}, iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
    for k, v in losses.items():
        message += '{0}: {1:.6f} '.format(k, v)

    print(message)  # print the message


def validate(checkpoints_path, best_params_file, params_features, lead_time, seednum, dataset_name, normalize_all_data,
             ckpt_id):
    with open(best_params_file, 'rb') as f:
        tuning_details = pickle.load(f)

    normalize_all_data = normalize_all_data == 'True'

    configuration = {
        "data_params": {
            "dataset_name": "regional_timeseries",
            "dataset_path": "./input_data/",
            "dataset_mode": dataset_name,
            "normalize_all_data_from_beginning": normalize_all_data,
            "variables": {
                "u10": True,
                "v10": True,
                "t2m": True,
                "resident_population": False,
                "resident_population_density": False,
                "lat": False,
                "long": False
            },
            "num_input_units": 20,
            "lead_time": lead_time,
            "train_ratio": 0.5,
            "val_ratio": 0.1
        },
        "train_dataset_params": {
            "dataset_name": "regional_timeseries",
            "dataset_path": "./input_data/",
            "num_input_units": 20,
            "lead_time": lead_time,
            "loader_params": {
                "batch_size": 32,
                "shuffle": False
            },
            "ratio": 0.5
        },
        "val_dataset_params": {
            "dataset_name": "regional_timeseries",
            "dataset_path": "./input_data/",
            "num_input_units": 20,
            "lead_time": lead_time,
            "loader_params": {
                "batch_size": 32,
                "shuffle": False
            },
            "ratio": 0.1
        },
        "model_params": {
            "seed": seednum,
            "model_name": "reilif",
            "dataset_mode": dataset_name,
            "is_train": False,
            "max_epochs": 2000,
            "lr": tuning_details.best_params['lr'],
            "export_path": "",
            "checkpoint_path": checkpoints_path,
            "load_checkpoint": ckpt_id,
            "lr_policy": "-",
            "lr_decay_iters": 20,
            "num_input_units": 20,
            "lead_time": lead_time,
            "hidden_dim_qk": tuning_details.best_params['hidden_dim_qk'],
            "hidden_dim_v": tuning_details.best_params['hidden_dim_v'],
            "hidden_dim_vsn_qk": tuning_details.best_params['hidden_dim_vsn_qk'],
            "hidden_dim_vsn_v": tuning_details.best_params['hidden_dim_vsn_v'],
            "lstm_n_layers": 1,
            "lstm_bidirectional": False,
            "dropout": 0.2,  # tuning_details.best_params['dropout'],
            "variable_selection": True,
            "attention_layer_normalization": True,
            "layer_normalization": True
        },
        "printout_freq": 100,
        "model_update_freq": 1,
        "trace_time": 100
    }

    if params_features == 'T-T-T-T-T':
        configuration['data_params']['variables']['resident_population'] = True
        configuration['data_params']['variables']['resident_population_density'] = True
    elif params_features == 'T-F-T-T-T':
        configuration['data_params']['variables']['resident_population'] = False
        configuration['data_params']['variables']['resident_population_density'] = False

    random.seed(configuration['model_params']['seed'])
    np.random.seed(configuration['model_params']['seed'])
    torch.manual_seed(configuration['model_params']['seed'])
    torch.cuda.manual_seed(configuration['model_params']['seed'])
    torch.backends.cudnn.deterministic = True

    print('Initializing dataset...')

    data, regions, varnames_to_ids = load_data_from_csv_to_dataframe(configuration['data_params'])
    train_set, val_set, test_set, scaler_ili = split_and_scale_data(data, configuration['data_params'], regions,
                                                                    varnames_to_ids)

    print('Initializing dataset...')
    test_dataset = create_dataset(test_set, regions, configuration['val_dataset_params'])
    test_dataset_size = len(test_dataset)
    print('The number of test samples = {0}'.format(test_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'], varnames_to_ids, scaler_ili)
    model.setup()
    model.eval()

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    for i, data in enumerate(test_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

    model.post_epoch_callback(configuration['model_params']['load_checkpoint'], params_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('checkpointsPath', help='path to the checkpoints')
    parser.add_argument('bestParamsFile', help='path to the optuna file')
    parser.add_argument('features', help='features/setup that are used in the following order Vars-Static-Vs-Aln-Ln, '
                                         'e.g., F-F-F-F-F')
    parser.add_argument('leadTime', help='lead time to predict')
    parser.add_argument('seed', help='seed number')
    parser.add_argument('dataset_mode', help='the name of the dataset')
    parser.add_argument('norm_all_data', help='if we follow Jung et al. '
                                              'setup regarding the normalization of the data or not')
    parser.add_argument('ckpt', help='which ckpt to use')

    args = parser.parse_args()
    print(optuna.__version__)
    validate(args.checkpointsPath, args.bestParamsFile, args.features, int(args.leadTime), int(args.seed),
             args.dataset_mode, args.norm_all_data, int(args.ckpt))

