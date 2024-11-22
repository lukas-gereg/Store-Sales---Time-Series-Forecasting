import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel



class DeepAR(BaseModel):
    def __init__(self, model_properties: dict):
        """
        DeepAR model for probabilistic time series forecasting.
        
        Args:
            model_properties (dict): Dictionary containing model properties:
                - input_size (int): Number of input features.
                - hidden_size (int): Number of hidden units in RNN.
                - num_layers (int): Number of RNN layers.
                - output_size (int): Number of prediction steps ahead (forecast horizon).
                - dropout (float): Dropout probability for regularization.
        """
        super(DeepAR, self).__init__(model_properties)
        
        self.input_size = model_properties.get("input_size", 10)
        self.hidden_size = model_properties.get("hidden_size", 64)
        self.num_layers = model_properties.get("num_layers", 2)
        self.output_size = model_properties.get("output_size", 1)
        self.dropout = model_properties.get("dropout", 0.1)
        
        self.init_model()

    def init_model(self):
        """
        Initialize RNN layers and fully connected layers for DeepAR.
        """
        # RNN layers (GRU can be replaced with LSTM if needed)
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        
        # Fully connected layers for prediction
        self.fc_mu = nn.Linear(self.hidden_size, self.output_size)  # Mean of the distribution
        self.fc_sigma = nn.Linear(self.hidden_size, self.output_size)  # Standard deviation

    def forward(self, x):
        """
        Forward pass through DeepAR.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: Mean (mu) and standard deviation (sigma) for the forecast.
        """
        # Initialize hidden states
        h0 = torch.zeros(
            self.num_layers, 
            x.size(0), 
            self.hidden_size
        ).to(x.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)  # Output shape: (batch_size, sequence_length, hidden_size)

        # Use the last hidden state for prediction
        last_hidden_state = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Calculate mean (mu) and standard deviation (sigma)
        mu = self.fc_mu(last_hidden_state)  # Shape: (batch_size, output_size)
        sigma = F.softplus(self.fc_sigma(last_hidden_state))  # Ensure non-negative sigma
        
        return mu, sigma



def negative_log_likelihood_loss(mu, sigma, target):
    """
    Negative Log Likelihood (NLL) loss for DeepAR.
    
    Args:
        mu (torch.Tensor): Predicted mean.
        sigma (torch.Tensor): Predicted standard deviation.
        target (torch.Tensor): True target values.
    
    Returns:
        torch.Tensor: NLL loss.
    """
    # Gaussian negative log likelihood
    loss = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)
    return loss.mean()
