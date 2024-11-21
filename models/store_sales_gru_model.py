import torch
import torch.nn as nn

from models.base_model import BaseModel


class StoreSalesGRUModel(BaseModel):
    def __init__(self, model_properties: dict) -> None:
        """
        Enhanced GRU network with improvements for time series prediction.
        
        Args:
            model_properties (dict): Dictionary containing model properties such as:
                - input_size (int): Number of input features.
                - hidden_size (int): Number of neurons in the GRU hidden layer.
                - num_layers (int): Number of GRU layers.
                - output_size (int): Number of output values (e.g., predicting 15 days).
                - dropout (float): Dropout probability for regularization.
        """
        super(StoreSalesGRUModel, self).__init__(model_properties)

        self.input_size = model_properties.get("input_size", 15)
        self.hidden_size = model_properties.get("hidden_size", 64)
        self.num_layers = model_properties.get("num_layers", 2)
        self.output_size = model_properties.get("output_size", 1)
        self.dropout = model_properties.get("dropout", 0.2)
        self.bidirectional = model_properties.get("bidirectional", False)

        self.init_model()

    def init_model(self) -> None:
        """
        Initialize the GRU model.
        """
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.output_size)
        self.relu = nn.ReLU()


    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the enhanced GRU network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predictions with shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers,  # Multiply by 2 for bidirectional
            x.size(0),
            self.hidden_size
        ).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, h0)

        # Use the last hidden state from both directions
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)

        # Normalize the hidden states
        #out = self.layer_norm(out)

        out = self.fc1(out) 

        # out = self.relu(out)

        # out = self.dropout(out)

        out = self.fc2(out)
        
        out = self.fc3(out)

        out = self.relu(out)

        # out = self.dropout(out)


        return out
