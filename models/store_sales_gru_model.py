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
        
        self.input_size = model_properties["input_size"]
        self.hidden_size = model_properties["hidden_size"]
        self.num_layers = model_properties["num_layers"]
        self.output_size = model_properties["output_size"]
        self.dropout = model_properties["dropout"]
        
        self.init_model()


    def init_model(self) -> None:
        """
        Model initialization.
        """
    
        self.gru = nn.GRU(
            self.input_size, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            dropout=self.dropout if self.num_layers > 1 else 0  # Dropout only if num_layers > 1
        )
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()


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
