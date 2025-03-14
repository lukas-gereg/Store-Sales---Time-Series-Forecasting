import torch
import pandas as pd

from datasets.base_dataset import BaseDataset


class SalesDataSet(BaseDataset):
    """
    Allows to load time series data, but is able to work normally given temporal length = 1
    """

    def __init__(self,
                 dataset: list[pd.DataFrame],
                 temporal_length: int,
                 capture_offset: int = 1,
                 window_offset: int = 1,
                 ):
        self.temporal_length = temporal_length
        self.labels = []
        self.files = []
        self.offset = capture_offset
        self.window_offset = window_offset

        self._set_data(dataset)

    def _set_data(self, datasets: list[pd.DataFrame]) -> None:
        """
        Instantiates data into Dataset class from list
        """

        for dataset in datasets:
            if 'sales' in dataset.columns:
                x_data = torch.tensor(dataset.drop(columns=['sales']).values, dtype=torch.float32)
                y_data = torch.tensor(dataset['sales'].values, dtype=torch.float32)
            else:
                x_data = torch.tensor(dataset.values, dtype=torch.float32)
                y_data = torch.tensor([0.0] * len(dataset), dtype=torch.float32)

            names = torch.tensor(dataset.index.tolist())

            self.files.append(dict({'x': x_data, 'y': y_data, 'names': names}))

    def __len__(self) -> int:
        """
        Returns dataset length
        """

        return sum([self.calculate_temporal_data_index_length(data['x']) for data in self.files])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, any, any]:
        """
        Return data, label and name of data with sequence from it behind index
        """

        data, data_name = self.find_x_by_index(idx)

        return data, self.find_y_by_index(idx), data_name

    def find_x_by_index(self, idx: int) -> tuple[torch.Tensor, any]:
        """
        Returns data behind index
        """

        return self.find_items(idx, 'x'), self.find_items(idx, 'names')

    def find_y_by_index(self, idx: int) -> any:
        """
        Returns label behind index
        """

        return self.find_items(idx, 'y')

    def find_items(self, idx, data_type: str):
        """
        Returns items behind index and type
        """

        data_dict, temporal_idx = self.get_data_dict_and_index(idx)

        data = data_dict[data_type]

        series_idx = (idx - temporal_idx) * self.window_offset

        return data[series_idx: series_idx + self.temporal_length: self.offset]


    def get_data_dict_and_index(self, idx):
        """
        :param idx: index of dataset sequence
        :return: whole stored data dictionary with combined length of previous data
        """

        temporal_idx = 0

        for data_dict in self.files:
            data = data_dict['x']

            if (temporal_idx + self.calculate_temporal_data_index_length(data)) > idx:
                return data_dict, temporal_idx
            else:
                temporal_idx += self.calculate_temporal_data_index_length(data)

    def calculate_temporal_data_index_length(self, temporal_data: list[any]) -> int:
        """
        :param temporal_data: data to calculate length of
        :return: number of sequences inside temporal data
        """

        data_part_length = int((len(temporal_data) - self.temporal_length) / self.window_offset) + 1

        return data_part_length if data_part_length >= 0 else 0