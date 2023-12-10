# Import necessary packages
from model_package.mclr import MCLR


class User:
    """
    Represents a user/client of the server with various attributes.

    Attributes:
        id (str): Unique identifier of the user.
        port_no (int): Communication port number.
        train_samples (int): Number of training samples.
        current_iteration (int): Current iteration in the training process.
        model (MCLR): Machine learning model associated with the user.
        accuracy (float): Accuracy of the user's model, initialized to 0.
        loss (float): Current loss of the user's model, initialized to 0.
    """
    
    def __init__(self, train_samples: int, port_no: int, id: str):
        """
        Initializes a new instance of User.

        Args:
            train_samples (int): The number of training samples the user has.
            port_no (int): The port number used by the client for communication.
            id (str): The unique identifier of the user.
        """
        self.id = id
        self.port_no = port_no
        self.train_samples = train_samples
        self.current_iteration = 0
        self.model = MCLR()
        self.accuracy = 0
        self.loss = 0