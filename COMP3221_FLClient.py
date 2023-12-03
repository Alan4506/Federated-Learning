# Import necessary packages
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader
import threading
import socket
import sys
from COMP3221_FLServer import MCLR


class Client:
    """
    Represents a client in a federated learning system, handling local data and model training.

    Attributes:
        client_id (str): A unique identifier for the client.
        id (int): Extracted numeric ID from the client_id.
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
        self.id = int(client_id[-1])
        self.server_port = 6000
        self.port = port
        self.host = "127.0.0.1"
        self.learning_rate = learning_rate
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = self.get_data()
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        
        if opt == "0":
            self.trainloader = DataLoader(self.train_data, self.train_samples)
        elif opt == "1":
            self.trainloader = DataLoader(self.train_data, 20)

        self.testloader = DataLoader(self.test_data, self.test_samples)
        
        self.model = MCLR()
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        log_path = os.path.join(".", self.client_id + "_log.txt")
        if os.path.exists(log_path):
            os.remove(log_path)
    
    
    def hand_shake(self):
        """
        Initiates a handshake with the server to establish a connection and sends client details.

        This method sends a hand-shaking packet containing the client's ID, port number,
        and number of training samples to the server.
        """
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((self.host, self.server_port))
        
        hand_shake_mess = f"hand-shaking packet\n{self.client_id}\n{self.port}\n{self.train_samples}"
        mess_data = bytes(hand_shake_mess, encoding='utf-8')
        connection.sendall(mess_data)
        connection.close()
        
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
        
        except socket.error as msg:
            print("Error during socket connection!")
            print(msg)
            
            
    def distributed_computation(self, conn: socket.socket):
        """
        Handles the distributed computation process after receiving the global model from the server.

        Args:
            conn (socket.socket): The socket connection through which data is received.
        """
        while True:
            received = ""
            while True:
                data = conn.recv(1024 * 16)
                if not data:
                    break
                received += str(data.decode('utf-8'))
            # received = str(data.decode('utf-8'))
            if (received == ""):
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
                os._exit(0)
        conn.close()
                
    def log_accuracy_and_loss(self, accuracy: float, loss: float):
        """
        Logs and prints the accuracy and loss of the client's model.

        Args:
            accuracy (float): The testing accuracy of the client's model.
            loss (float): The training loss of the client's model.
        """
        with open(self.client_id + "_log.txt", "a") as log:
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
                s.close()   
                
        except socket.error as msg:
            print("Error during local model sending!")
            print(msg)
                
                       
    def get_data(self) -> tuple:
        """
        Retrieves and parses the client's training and testing datasets from JSON files into Tensor objects.

        Returns:
            tuple: Tensors representing training and testing images and labels, and sample counts.
        """
        train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(self.id) + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(self.id) + ".json")
        train_data = {}
        test_data = {}

        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data'])

        X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
        X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
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
            
    def train(self, epochs: int) -> list:
        """
        Trains the local model for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.

        Returns:
            list: The final loss of the model after training, converted to a list.
        """
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data.tolist()
    
    def test(self) -> float:
        """
        Tests the trained model on the test dataset to calculate the accuracy.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) / y.shape[0]).item()
        return test_acc

def main():
    """
    The main function that initializes a client instance and starts the training and testing process.

    This function parses command line arguments for client ID, port number, and optimization method.
    It validates the inputs and starts the handshaking and model receiving processes in separate threads.
    """
    if len(sys.argv) != 4:
        print("Require 4 command line arguments!")
        return
    
    client_id = sys.argv[1]
    port_no = sys.argv[2]
    opt_method = sys.argv[3]
    learning_rate = 0.01
    
    try:
        id = int(client_id[-1])
    except:
        print("Wrong client id!")
        return
    
    try:
        port_no = int(port_no)
    except:
        print("Wrong port No!")
        return
    
    if opt_method != "0" and opt_method != "1":
        print("Wrong optimization method!")
        return
    
    client = Client(client_id, learning_rate, port_no, opt_method)
    
    client.hand_shake()
    client.model_receiving()
    
# The starting point of the script. 
if __name__ == "__main__":
    main()
