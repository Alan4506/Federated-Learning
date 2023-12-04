# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import random
import time
import matplotlib.pyplot as plt
import threading
import socket


class MCLR(nn.Module):
    """
    The model used for the MNIST task: a simple multinominal logistic regression model
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


class Server:
    """
    Represents the server in a federated learning setup, managing the global model and communication with clients.

    Attributes:
        init_time (float): Timestamp of when the server was initialized.
        port_no (int): Port number used for server-client communication.
        sub_client (int): A number that determines if the clients subsampling is enabled (1) or not (0).
        host_no (str): Host address.
        users (list of User): List of User objects representing clients connected to the server.
        global_model (MCLR): The global model being trained.
        current_iteration (int): The current iteration of training.
        loss (list of float): List of loss values from each training round.
        accuracy (list of float): List of accuracy values from each training round.
        random_users (list of User): Random subset of users selected for aggregation.
        new_users (list of User): New users that connect to the server.
        iteration_time (float): Timestamp of the start of the current iteration.
    """

    def __init__(self, port_no: int, sub_client: int):
        """
        Initializes the Server with the given port number and sub_client count.
        
        Args:
            port_no (int): The port number on which the server will listen.
            sub_client (int): A flag that determines whether the clients subsampling is enabled or not.
        """
        self.init_time = time.time()
        self.port_no = port_no
        self.sub_client = sub_client
        self.host_no = "127.0.0.1"
        self.users = []
        self.global_model = MCLR()
        self.current_iteration = 1
        self.loss = []
        self.accuracy = []
        self.random_users = []
        self.new_users = []
        self.iteration_time  = 0

    def check_user_existence(self, client_id: str) -> bool:
        """
        Checks if a user with the specified client ID already exists in the server's user list.

        Args:
            client_id (str): The client ID to check for existence.

        Returns:
            bool: True if a user with the given client ID exists, False otherwise.
        """
        for user in self.users:
            if user.id == client_id:
                return True
        return False

    def iterate(self):
        """
        Runs the training process for 100 global communication rounds. Clients are stopped after 100 rounds.

        This method handles the main training loop, broadcasting the global model, receiving updates from clients,
        aggregating parameters, and updating the global model.
        """
        time.sleep(30)
        while self.current_iteration <= 100:

            self.iteration_time = time.time()

            # Broadcast global model to all clients
            print("Broadcasting new global model")
            self.broadcast()
        
            print("Global Iteration " + str(self.current_iteration) + ":")
            print("Total Number of clients: " + str(len(self.users)))

            # after received all local packets
            while True:
                time.sleep(0.001)
                self.drop_dead_clients()
                if (self.check_all_received() == True):
                    break
            
            # Evaluate the global model across all clients
            avg_accuracy, avg_loss = self.evaluate()
            self.accuracy.append(avg_accuracy)
            self.loss.append(avg_loss)

            print("Average accuracy of all clients: {:.4f}".format(avg_accuracy))
            print("Average loss of all clients: {:.4f}".format(avg_loss))
            
            # Aggregate all clients model to obtain new global model 
            self.aggregate_parameters()
            print("Aggregating new global model\n")

            self.current_iteration += 1

            for user in self.new_users:
                self.users.append(user)
            self.new_users = []

        self.stop_clients()

    def aggregate_parameters(self) -> MCLR:
        """
        Aggregates parameters from the client models to update the global model.

        If `sub_client` is 0, aggregates from all users, otherwise selects a random subset of users.
        Updates the global model's parameters by averaging the client models weighted by their number of samples.

        Returns:
            MCLR: The updated global model after aggregation.
        """
        # Clear global model first
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        if (len(self.users) == 0):
            return
        
        if (self.sub_client == 0):
            total_samples = 0
            for user in self.users:
                total_samples += user.train_samples
            for user in self.users:
                for server_param, user_param in zip(self.global_model.parameters(), user.model.parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_samples
            return self.global_model

        else:
            if (len(self.users) == 1):
                self.random_users = [self.users[0]]
            else:
                candidates = range(0, len(self.users))
                values = random.sample(candidates, 2)
                self.random_users = [self.users[values[0]], self.users[values[1]]]

            total_samples = 0
            for user in self.random_users:
                total_samples += user.train_samples
            for user in self.random_users:
                for server_param, user_param in zip(self.global_model.parameters(), user.model.parameters()):
                    server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_samples
            return self.global_model

    def evaluate(self) -> tuple:
        """
        Calculates and returns the average testing accuracy and average training loss of all users.

        Returns:
            tuple: A tuple containing the average accuracy and average loss across all users.
        """
        total_accurancy = 0
        total_loss = 0
        if (len(self.users) == 0):
            return 0, 0
        for user in self.users:
            total_accurancy += user.accuracy
            total_loss += user.loss
        return total_accurancy / len(self.users), total_loss / len(self.users)

    def receive(self):
        """
        Listens for and receives packets from clients, starting a new thread for handling each connection.

        This method is responsible for handling the network communication, accepting incoming connections,
        and delegating the packet handling to `data_receiving` method in separate threads.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind((self.host_no, self.port_no))
                server.listen(10)

                while True:
                    c, addr = server.accept()
                    conn = threading.Thread(target=self.data_receiving, args=(c,))
                    conn.start()

                server.close()

        except:
            print("server error")

    def data_receiving(self, client: socket.socket):
        """
        Handles the packets received from a client socket.

        This method processes incoming data, registers new users, and updates existing user data
        with new training parameters received from the client.

        Args:
            client (socket.socket): The client socket from which data is received.
        """
        while True:
            received = ""
            while True:
                data = client.recv(1024 * 16)
                if not data:
                    break
                received += str(data.decode('utf-8'))

            if (received == ""):
                break

            lines = received.split("\n")
            if (lines[0] == "hand-shaking packet"):
                if (self.check_user_existence(lines[1]) == False):
                    user = User(int(lines[3]), int(lines[2]), lines[1])
                    if (time.time() - self.init_time <= 30):
                        self.users.append(user)
                    else:
                        self.new_users.append(user)

            else:
                id = lines[1]
                print("Getting local model from client " + id[-1])
                acc = lines[2]
                loss = lines[3]
                param1_str = lines[4]
                param2_str  = lines[5]
                for user in self.users:
                    if id == user.id:
                        user.current_iteration = self.current_iteration
                        param1, param2 = user.model.parameters()
                        param1.data = torch.tensor(json.loads(param1_str))
                        param2.data = torch.tensor(json.loads(param2_str))
                        user.accuracy = float(acc)
                        user.loss = float(loss)

        client.close()

    def broadcast(self):
        """
        Broadcasts the current global model's parameters to all connected clients.

        This method serializes the global model's parameters and sends them over a socket connection
        to each client. It handles the potential failure of clients to receive the model by catching
        exceptions and printing an error message.
        """
        param1, param2 = self.global_model.parameters()
        message = "broadcast packet\n"
        message += str(param1.data.tolist()) + "\n"
        message += str(param2.data.tolist()) + "\n"

        for user in self.users:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host_no, user.port_no)) 
                    message_encrypt = bytes(message, encoding= 'utf-8')
                    s.sendall(message_encrypt)
                    s.close()   
            except:
                print("Broadcast the global model to " + user.id + " failed. This client may have failed")
                pass
    
    def stop_clients(self):
        """
        Sends a message to all clients to stop their training process.

        This method iterates through all users and sends a 'stop packet' message
        to signal the end of the training. It handles exceptions in case the client has failed or disconnected.
        """
        for user in self.users:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.host_no, user.port_no)) 
                    message = "stop packet"
                    message_encrypt = bytes(message, encoding= 'utf-8')
                    s.sendall(message_encrypt)
                    s.close()   
            except:
                print("stop clients error")
                pass

    def drop_dead_clients(self):
        """
        Removes clients from the server's user list if they have not updated for the current iteration.

        This method checks if the clients have failed to send their updated model within a specific time frame
        and removes them from the list of active clients to maintain an updated and active client pool.
        """
        users_to_remove = []
        for user in self.users:
            if user.current_iteration != self.current_iteration and time.time() - self.iteration_time > 15:
                users_to_remove.append(user)
        for user in users_to_remove:
            self.users.remove(user)
    
    def check_all_received(self) -> bool:
        """
        Checks if all clients have sent their local model packets for the current iteration.

        Returns:
            bool: True if all clients' data has been received for the current iteration, False otherwise.
        """
        for user in self.users:
            if user.current_iteration != self.current_iteration:
                return False
        return True

    def plot_loss(self):
        """
        Plots the training loss of the global model over the iterations.

        This method creates a matplotlib figure to visualize the training loss progression
        throughout the federated learning process.
        """
        plt.figure(1,figsize=(5, 5))
        plt.plot(self.loss, label="FedAvg", linewidth  = 1)
        plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
        plt.ylabel('Training Loss')
        plt.xlabel('Global rounds')
        plt.show()

    def plot_accuracy(self):
        """
        Plots the testing accuracy of the global model over the iterations.

        This method creates a matplotlib figure to visualize the testing accuracy progression
        throughout the federated learning process.
        """
        plt.figure(1,figsize=(5, 5))
        plt.plot(self.accuracy, label="FedAvg", linewidth  = 1)
        plt.ylim([0.8,  0.99])
        plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
        plt.ylabel('Testing Acc')
        plt.xlabel('Global rounds')
        plt.show()

    def run(self):
        """
        Starts the server and initiates the federated learning process.

        This method sets up threads for receiving data from clients and for iterating through
        training rounds, then plots the loss and accuracy after the training is complete.
        """
        t1 = threading.Thread(target=self.receive)
        t1.daemon = True
        t2 = threading.Thread(target=self.iterate)
        t2.start()
        t1.start()
        t2.join()
        self.plot_loss()
        self.plot_accuracy()

# This block is the starting point of the script. 
# It parses command line arguments for the server's port number and sub-client count, 
# initializes the server, and starts the federated learning process.
if __name__ == '__main__':
    port_no = int(sys.argv[1])
    sub_client = int(sys.argv[2])
    server = Server(port_no, sub_client)
    server.run()
