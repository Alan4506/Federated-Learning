# Learning Objectives
![image](https://github.com/Alan4506/Federated-Learning/assets/62124408/c61071e0-eb33-4057-a627-a3517ca23a64)

The task is to implement a simple Federated Learning (FL) system including five clients in total and one server for aggregation in Fig. 1. Each client has its own data used for training its local model and then contributes its local model to the server through the socket in order to build the global model. It is noted that we can simulate the FL system on a single machine by opening different terminals (one each for every client and one for server) on the same machine (use "localhost").

# Federated Learning Algorithm
![image](https://github.com/Alan4506/Federated-Learning/assets/62124408/846f4bce-22fb-435c-9ebd-f0e01f99c9c6)

In this assignment, we implement FedAvg, presented in Alg. 2.1. T is the total number of global communication rounds between clients and the server. While w_t is the global model at iteration t, w_{t+1}^k is the local model of client k at iteration t+1. To obtain the local model, clients can use both Gradient Descent (GD) or Mini-Batch Gradient Descent (Mini-Batch GD) as the optimization methods.

# Dataset and classification model
We consider MNIST, a handwritten digits dataset including 70,000 samples belonging to 10 classes (from 0 to 9) for this assignment. In order to capture the heterogeneous and non-independent and identically distributed (non. i.i.d) settings in FL, the original dataset has been distributed to K = 5 clients where each client has different data sizes and has maximum of 3 over 10 classes. 

The dataset is located in folder named "FLdata". Each client has 2 data files (training data and testing data) stored in 2 json files. For example: traing and testing data for Client 1 are "mnist_train_client1.json" and "mnist_test_client1.json", respectively.

![image](https://github.com/Alan4506/Federated-Learning/assets/62124408/10dc5581-ad1a-4be7-9019-1e61ffa577a2)

As MNIST is used for a classification problem, we can choose any classification methods such as multinominal logistic regression, DNN, CNN, etc... as the local model for all clients. However, all clients have to have the same kind of classification model. To simplify implementation, we use the multinominal logistic regression.

#  Program structure
There are 2 main programs implemented: one for clients and one for the server. The server program has to be started before starting client programs.
##  Server

The server program should be named as `COMP3221_FLServer.py` and accepts the following command-line arguments:

```
python COMP3221_FLServer.py <Port-Server> <Sub-client>
```

For example: 

```
python COMP3221_FLServer.py 6000 1
```

- `Port-Server`: is the port number of the server used for listening model packets from clients and it is fixed to 6000.

- `Sub-client`: is a flag to enable clients subsampling. (0 means M = K then the server will aggregate all clients model, 1 means M = 2 then the server only aggregate randomly 2 clients over 5 clients).

Following Alg. 2.1, initially, the server randomly generates the global model w_0 and listens for hand-shaking messages from new coming clients. Whenever the server receives the handshaking message from a new client, it will add this client to a list (client’s list) that contains all clients in the FL system. The hand-shaking message includes the client’s data size and id.

After receiving the first hand-shaking message of one client, the server will wait for 30s to listen to new other coming clients for registration (this process only happens once when the server starts). The server then broadcasts the global model to all registered clients and then waits for all clients to send new local models for aggregation. You are free to define the exact format of the model packets and hand-shaking messages.

After collecting all local models from all clients, the server aggregates all client’s models (or subset M = 2 clients depends on the setting) to form a new global model and broadcasts the new global model to all registered clients. This process is one global comunication round. In this assigment, FL system will run T = 100 global communication rounds. After 100 global communication rounds, server will broadcast a message to stop training process to all clients and then stops training process. For each global round, server will print out the following output to the terminal:

```
Global Iteration 10:
Total Number of clients: 5
Getting local model from client 1
Getting local model from client 2
Getting local model from client 5
Getting local model from client 4
Getting local model from client 3
Aggregating new global model
Broadcasting new global model
```

If there is a new client coming for registration after the server has finished the initialization, the server will add this client to the current client’s list and broadcast the global model in the next global communication round.
##  Client
The client program should be named as `COMP3221_FLClient.py` and accepts the following command line arguments:

```
python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>
```

For example: 

```
python COMP3221_FLClient.py client1 6001 1
```

- `Client-id`:  is ID of a client in a federated learning network and is indexed as following client1, client2, client3, client4, and client5.

- `Sub-client` is the port number of a client receving the model packets from the server. The port number is integer indexed from 6001 and increased by one for each client. For example the port number from client 1 to client 5 are from 6001 to 6005.

- `Opt-Method`: is an optimization method to obtain local model (0 is for GD and 1 is for Mini-Batch GD).

Upon initialization, each client loads its own data, sends the hand-shaking message to the server for registration, and waits for the server to broadcast the global mode. On receiving the global model packets, the client uses this model to evaluate the local test data. It also logs the training loss and accuracy of the global model at each communication round to a file named `client1_log.txt` (for evaluation purpose) and prints out to the terminal, for example:

```
I am client 1
Receving new global model
Training loss: 0.01
Testing acurancy: 98%
Local training...
Sending new local model
```

After that, the client uses the global model for continuing the training process to create a new local model. The local training process can be finished in E = 2 local iterations using GD or Mini-Batch GD. The client then sends the new local model to the server and waits for receiving the new global model from the server. The batch-size for Mini-Batch GD is set to 20.
##  Evaluation
To evaluate the performance of the global model across all clients at each global communication round, we take the average of training loss and testing accuracy across all clients. This is done after all clients and the server finished the training process.
