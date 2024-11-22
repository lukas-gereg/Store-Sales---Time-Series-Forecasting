import torch
from torch.utils.data import Dataset



# Custom Dataset for Sequence Data
class SalesDataset(Dataset):
    def __init__(self, df, sequence_length):
        """
        Initializes the dataset for sequence-based data.

        Args:
            df (pd.DataFrame): DataFrame containing features and (optionally) the target.
            sequence_length (int): Number of past days to use for predictions.
        """
        self.sequence_length = sequence_length

        # Check if 'sales' column exists and handle accordingly
        if 'sales' in df.columns:
            self.features = df.drop(columns=['sales']).values
            self.targets = df['sales'].values
        else:
            self.features = df.values
            self.targets = None

        self.length = len(df) - sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        if self.targets is not None:
            y = self.targets[idx + self.sequence_length]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32)
        
