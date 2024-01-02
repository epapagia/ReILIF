from datasets.base_dataset import BaseDataset
import torch


class TimeSeriesDataset(BaseDataset):
    """Wrapper class of Dataset class that performs multi-threaded data loading
        according to the configuration.
    """

    def __init__(self, dataset_x: dict, dataset_y: dict, regions_list: list,
                 configuration: dict):
        super().__init__(dataset_x, dataset_y, regions_list, configuration)
        # samplesX has dims [Number of dataset examples, T units, number of variables, N states]
        self.samplesX = torch.zeros(dataset_x[regions_list[0]].shape[0], dataset_x[regions_list[0]].shape[1], dataset_x[regions_list[0]].shape[2], len(dataset_x))
        self.samplesY = torch.zeros(dataset_x[regions_list[0]].shape[0], len(regions_list))

        for sample_id in range(dataset_x[regions_list[0]].shape[0]):
            # a sample in each batch has dims [T units, number of variables, N states]
            for state_id, state in enumerate(regions_list):
                # transpose to get [T units, number of variables, 1], i.e. T units is the seq len
                # print('dataset_x[state][sample_id, :].shape', dataset_x[state][sample_id, :].shape)
                self.samplesX[sample_id, :, :, state_id] = dataset_x[state][sample_id, :, :]
                self.samplesY[sample_id, state_id] = dataset_y[state][sample_id][0]

    def __len__(self):
        return self.samplesX.shape[0]

    def __getitem__(self, idx: int):
        return self.samplesX[idx, :, :], self.samplesY[idx, :]

