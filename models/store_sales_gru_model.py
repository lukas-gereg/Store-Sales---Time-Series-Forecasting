import torch
import torch.nn as nn

from models.base_model import BaseModel


class StoreSalesGRUModel(BaseModel):
    def __init__(self, model_properties: dict) -> None:
        """
        Initialization of the GRU network.
        
        Args:
            model_properties (dict): Dictionary containing model properties such as:
                - input_size (int): Number of input features.
                - hidden_size (int): Number of neurons in the GRU hidden layer.
                - num_layers (int): Number of GRU layers.
                - output_size (int): Number of output values (15 days).
                - dropout (float): Dropout probability for regularization.
        """

        super(StoreSalesGRUModel, self).__init__(model_properties)
        
        # Initializing with default values if not provided in model_properties
        self.input_size = model_properties.get(key="input_size", default=15)    # Default: 15 input features
        self.hidden_size = model_properties.get(key="hidden_size", default=64)  # Default: 64 hidden neurons
        self.num_layers = model_properties.get(key="num_layers", default=2)     # Default: 2 GRU layers
        self.output_size = model_properties.get(key="output_size", default=15)  # Default: prediction for 15 days
        self.dropout = model_properties.get(key="dropout", default=0.0)         # Default: no dropout
        
        self.init_model()


    def init_model(self) -> None:
        """
        Model initialization.
        """
    
        self.gru = nn.GRU(
            input_size = self.input_size, 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers, 
            batch_first = True, 
            dropout = self.dropout if self.num_layers > 1 else 0.0  # Dropout only if num_layers > 1
        )
        
        self.fc = nn.Linear(in_features = self.hidden_size, out_features = self.output_size)
        self.relu = nn.ReLU(inplace = True)


    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Predictions with shape (batch_size, output_size)
        """

        # Initialize the hidden state (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(
            self.num_layers, 
            x.size(0), 
            self.hidden_size
        ).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Use the last hidden layer output dimensions
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        out = self.fc(out)  # (batch_size, output_size)
        out = self.relu(out) # ReLU for non negative outputs

        return out
