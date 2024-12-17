import os
import torch
import pandas as pd
import torch.utils.data as torch_utils

from utils.rmsle import RMSLE
from datasets.sales_dataset import SalesDataSet
from data_preprocessing import DataPreprocessing
from utils.results_generation import ResultGeneration
from models.store_sales_gru_model import StoreSalesGRUModel

if __name__ == "__main__":
    debug = False
    folds = 5
    random_seed = 42

    scheduler = None
    early_stopping = 25

    BATCH_SIZE = 512

    epochs = 10000
    lr = 0.0001

    temporal_length = 14
    window_offset = 1
    output_size = 1
    hidden_size = 256

    # dataframe = DataPreprocessing.create_dataset(os.path.join(os.getcwd(), 'data', 'train.csv'))
    # dataframe.to_csv(os.path.join(os.getcwd(), 'output_train.csv'))
    # dataframe = DataPreprocessing.create_dataset(os.path.join(os.getcwd(), 'data', 'test.csv'))
    # dataframe.to_csv(os.path.join(os.getcwd(), 'output_test.csv'))

    dataframe = pd.read_csv(os.path.join(os.getcwd(), 'output_train.csv'), index_col="id")
    _, encoders = DataPreprocessing.preprocess_dataset(dataframe)
    dataframe = pd.read_csv(os.path.join(os.getcwd(), 'output_test.csv'), index_col="id")
    dataframe, encoders = DataPreprocessing.preprocess_dataset(dataframe, encoders)

    dataframes = DataPreprocessing.group_by_gap(dataframe, encoders)
    # DataPreprocessing.plot_seasonality(dataframes, ['sales'])

    base_dataset = SalesDataSet(dataframes, temporal_length, window_offset=window_offset)

    df = next(iter(base_dataset.files))['x']

    model_properties = {
        'input_size': df.shape[1],
        'hidden_size': hidden_size,
        # 'num_layers': 8,
        'output_size': output_size,
        # 'dropout': 0.1
    }

    model = StoreSalesGRUModel(model_properties)
    model.load_weights(os.path.join(os.getcwd(), 'model_params', 'folds', 'run-confused-donkey-8',
                                    'run-confused-donkey-8-fold-0-params.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = RMSLE()

    dataloader = torch_utils.DataLoader(base_dataset, batch_size=BATCH_SIZE)

    prediction_df = ResultGeneration(debug)(model, dataloader, loss, device)
    prediction_df.sort_index(ascending=True, inplace=True)

    prediction_df.to_csv('submission.csv', index_label='id')