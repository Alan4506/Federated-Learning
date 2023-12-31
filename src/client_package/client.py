
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader
import threading
import socket
from model_package.mclr import MCLR

class Client:
    """
    Represents a client in a federated learning system, handling local data and model training.

    Attributes:
        client_id (str): A unique identifier for the client.
        server_port (int): The port number of the server for communication.
        port (int): The local port for client communication.
        host (str): Hostname or IP address of the server.
        learning_rate (float): Learning rate for the SGD optimizer.
        X_train, y_train, X_test, y_test (array-like): Training and testing datasets.
        train_samples, test_samples (int): Number of samples in the training and testing datasets.
        trainloader, testloader (DataLoader): DataLoaders for batching training and testing data.
        model (MCLR): The local model for training.
        loss (Loss): The loss function.
        optimizer (Optimizer): The optimizer for training the model.
        log_path (str): The path of the log file
    """

    def __init__(self, client_id: str, learning_rate: float, port: int, opt: str):
        """
        Initializes a new Client instance with the given parameters, datasets, and machine learning model.

        Args:
            client_id (str): The unique identifier for the client.
            learning_rate (float): The learning rate to be used for the SGD optimizer.
            port (int): The local port for client communication.
            opt (str): Option to determine the optimization method; '0' for GD, '1' for mini-batch size GD.
        """
        self.client_id = client_id
        self.server_port = 6000
        self.port = port
        self.host = "127.0.0.1"
        self.learning_rate = learning_rate
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = self.load_data()

        self.train_data = list(zip(self.X_train, self.y_train))
        self.test_data = list(zip(self.X_test, self.y_test))
        
        if opt == "0":
            self.trainloader = DataLoader(self.train_data, self.train_samples)
        elif opt == "1":
            self.trainloader = DataLoader(self.train_data, 20)

        self.testloader = DataLoader(self.test_data, self.test_samples)
        
        self.model = MCLR()
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        self.log_path = os.path.join("..", "logs", self.client_id + "_log.txt")
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
    
    
    def register(self) -> bool:
        """
        First communicate with the server to establish a connection and sends client details.

        This method sends a client register packet containing the client's ID, port number,
        and number of training samples to the server.

        Returns:
            bool: True if registration is successful, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.connect((self.host, self.server_port))

                register_mess = f"client register packet\n{self.client_id}\n{self.port}\n{self.train_samples}"
                mess_data = bytes(register_mess, encoding='utf-8')
                connection.sendall(mess_data)

            return True  # Registration successful

        except Exception as e:
            print("Error during client registration:")
            print(e)
            return False  # Registration failed
        
    def model_receiving(self):
        """
        Sets up a socket connection to receive the global model from the server.

        This method establishes a socket connection and listens for incoming connections.
        Upon accepting a connection, it starts a new thread for distributed computation.
        If there's an error in the socket connection, it prints an error message.
        """
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as receiver:
                receiver.bind((self.host, self.port))
                receiver.listen(5)
                while True:
                    conn, addr = receiver.accept()
                    computation_thread = threading.Thread(target=self.distributed_computation, args=(conn,))
                    computation_thread.setDaemon(True)
                    computation_thread.start()
        
        except Exception as e:
            print("Error during socket connection!")
            print(e)
            
    def distributed_computation(self, conn: socket.socket):
        """
        Handles the distributed computation process after receiving the global model from the server.

        Args:
            conn (socket.socket): The socket connection through which data is received.
        """
        try:
            while True:
                received = ""
                while True:
                    data = conn.recv(1024 * 16)
                    if not data:
                        break
                    received += data.decode('utf-8')
                if received == "":
                    break
                received_ls = received.strip().split("\n")

                if received_ls[0] == "broadcast packet":
                    param1 = json.loads(received_ls[1])
                    param2 = json.loads(received_ls[2])
                    self.set_parameters(param1, param2)
                    
                    accuracy = self.test()
                    loss = self.train(2)
                    
                    sending_thread = threading.Thread(target=self.send_local_model, args=(accuracy, loss))
                    sending_thread.start()
                    
                    self.log_accuracy_and_loss(accuracy, loss)
                
                elif received_ls[0] == "stop packet":
                    print("All training finished.\n")
                    os._exit(0)
        
        except Exception as e:
            print(f"Error in distributed computation: {e}")

        finally:
            conn.close()

                
    def log_accuracy_and_loss(self, accuracy: float, loss: float):
        """
        Logs and prints the accuracy and loss of the client's model.

        Args:
            accuracy (float): The testing accuracy of the client's model.
            loss (float): The training loss of the client's model.
        """
        with open(self.log_path, "a") as log:
            message = f"I am {self.client_id[:-1]} {self.client_id[-1]}\n"
            message += f"Receiving new global model\nTraining loss: {loss:.4f}\n" 
            message += f"Testing accuracy: {accuracy * 100:.4f}%\n"
            message += "Local training...\nSending new local model\n\n"
            
            log.write(message)
            print(message)
            
    def send_local_model(self, accuracy: float, loss: float):
        """
        Sends the local model parameters along with accuracy and loss to the server.

        Args:
            accuracy (float): The testing accuracy of the client's model.
            loss (float): The training loss of the client's model.
        """
        param1, param2 = self.model.parameters()
        message = f"local model packet\n{self.client_id}\n{str(accuracy)}\n{str(loss)}\n"
        message += str(param1.data.tolist()) + "\n"
        message += str(param2.data.tolist()) + "\n"
            
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.server_port)) 
                mess = bytes(message, encoding= 'utf-8')
                s.sendall(mess)
                
        except Exception as e:
            print("Error during local model sending!")
            print(e)
                
                       
    def load_data(self) -> tuple:
        """
        Retrieves and parses the client's training and testing datasets from JSON files into Tensor objects.

        Returns:
            tuple: Tensors representing training and testing images and labels, and sample counts.
        """
        train_path = os.path.join("..", "data", "FLdata", "train", "mnist_train_client" + self.client_id[-1] + ".json")
        test_path = os.path.join("..", "data", "FLdata", "test", "mnist_test_client" + self.client_id[-1] + ".json")
        
        with open(train_path, "r") as f_train:
            train_data = json.load(f_train)['user_data']

        with open(test_path, "r") as f_test:
            test_data = json.load(f_test)['user_data']

        X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
        X_train = torch.Tensor(X_train).reshape(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).reshape(-1, 1, 28, 28).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_samples, test_samples = len(y_train), len(y_test)
        return X_train, y_train, X_test, y_test, train_samples, test_samples

    def set_parameters(self, param1_str: str, param2_str: str):
        """
        Updates the local model with the global model parameters received from the server.

        Args:
            param1_str (str): Serialized parameters for the first layer.
            param2_str (str): Serialized parameters for the second layer.
        """
        param1, param2 = self.model.parameters()
        param1.data = torch.tensor(param1_str)
        param2.data = torch.tensor(param2_str)
            
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            
    def train(self, epochs: int) -> float:
        """
        Trains the local model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.

        Returns:
            float: The final loss value of the model after training.
        """
        self.model.train()
        for epoch in range(1, epochs + 1):
            for X, y in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                
        return loss.item()
    
    def test(self) -> float:
        """
        Tests the trained model on the test dataset to calculate the accuracy.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.model.eval()
        test_acc = 0
        for X, y in self.testloader:
            output = self.model(X)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) / y.shape[0]).item()

        return test_acc