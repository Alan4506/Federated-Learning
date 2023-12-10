# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLR(nn.Module):
    """
    The model used for the MNIST task: a simple multinomial logistic regression
    """

    def __init__(self):
        """
        Initializes the MCLR model with a single fully connected layer.
        """
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc1.weight.data=torch.randn(self.fc1.weight.size())*.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MCLR model.
        
        Args:
            x (torch.Tensor): The input tensor containing the data.
            
        Returns:
            torch.Tensor: The output tensor after applying the log softmax function.
        """
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output