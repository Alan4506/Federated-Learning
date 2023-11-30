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
# Implementation
## Server
The server keeps a global model that is randomly generated initially. It will first start one thread (t1) for receiving packets from clients and another thread (t2) for running100 global communication rounds. The class User represents the client/user of theserver, storing clients’ port number, the iteration it is current at, the latest testingaccuracy and training loss, etc. After the server starts up, t2 sleeps for 30 seconds andt1 may receive some hand-shaking packets from clients. New clients will be addedtothe user list of the server, with their numbers of train samples, port numbers, and client ids recorded.

After the 30-second sleep, t1 will receive and handle hand-shaking fromthe newclient and local model packets from existing clients. By hand-shaking packets, newclients will be added to the user list at the end of the current communication round. And for local model packets, they include the client id, the testing accuracy of the global model that the client calculates, the training loss of the client’s new model, and the parameters of the client’s new model, recorded for each client respectively by the server. What is more, that client’s current_iteration attribute will be set to be the same as the server’s current_iteration to keep the cohesion.

On the other hand, t2 will broadcast its global model to all clients in the user list bysending broadcast packets, and detect if there is any dead client and whether it has received local model packets from every client. No response from a client for more than15s will be treated as the death of the client, and it will be removed fromthe user list. After all clients’ packets are received, the server will calculate and record the averagetraining loss and average testing accuracy of all clients. Then it will clear the global model and aggregate the models of clients. The number of clients it aggregates depends on the Sub_client flag. If the flag is 0, all 5 clients will be aggregated. If the flagis 1, only 2 clients will be selected randomly and aggregated. After that, those newlyregistered users of the current round will be added to the user list, and a newcommunication round will start.

After 100 global communication rounds end, the server will send a “stop packet” toeach client in the user list to tell the clients to finish their training processes, and thethread (t2) which executes the “iterate(self)” method ends. Since t1 is a daemon threadand there are no non-daemon threads, t1 will also end. After t1 and t2 finish, eachglobal communication round’s average training loss and average testing accuracy areplotted for further analysis.
## Client
As a client is initialized by a terminal running the COMP3221_FLClient.py with requiredarguments, it will first load the data from the corresponding MNIST dataset bytransferring the raw JSON files into dictionaries and creating the torch.Tensor objects based on them, and further loading these Tensor objects using the Dataloader. It will also do some basic preparation like setting up the model MCLR, the loss calculator torch.nn.NLLLoss and the optimizer torch.optim.SGD to support different types of gradient descent machine learning algorithms, and of course delete the logs generatedin the last running. Meanwhile, it will send hand-shaking messages to the server implying that the client is set up. After 30s, it will first test the global model receivedfrom the server in this round and record the accuracy. Then it will start to use GDor mini-batch GD to train the model on each batch for 2 epochs and return the loss of thecurrent training eventually. After the calculation when the 2 epochs end, the client will send its client id, the accuracy, the loss and the local model to the server, wrapped inastring and concatenated by new line characters. The server will then aggregate all clients’ local models and send the updated global model to the client to do a newglobal communication round. Once it reaches 100 rounds, a stop message will bereceived from the server and the client thus stops its thread.
# Run the Programs
To run the program, go to the 500170629_500089387_COMP3221_FLCode directory, and open 6 terminals.

The server must get started first. In the first terminal, run "python COMP3221_FLServer.py 6000 0" to disable clients subsampling or run "python COMP3221_FLServer.py 6000 1" to enable clients subsampling.

Then within 30 seconds, run 5 commands in the last 5 terminals.

In the second terminal, run "python COMP3221_FLClient.py client1 6001 0" to use GD as the client's optimization method or run "python COMP3221_FLClient.py client1 6001 1" to use Mini-Batch GD as the client's optimization method.

In the third terminal, run "python COMP3221_FLClient.py client2 6002 0" to use GD as the client's optimization method or run "python COMP3221_FLClient.py client2 6002 1" to use Mini-Batch GD as the client's optimization method.

In the fourth terminal, run "python COMP3221_FLClient.py client3 6003 0" to use GD as the client's optimization method or run "python COMP3221_FLClient.py client3 6003 1" to use Mini-Batch GD as the client's optimization method.

In the fifth terminal, run "python COMP3221_FLClient.py client4 6004 0" to use GD as the client's optimization method or run "python COMP3221_FLClient.py client4 6004 1" to use Mini-Batch GD as the client's optimization method.

In the sixth terminal, run "python COMP3221_FLClient.py client5 6005 0" to use GD as the client's optimization method or run "python COMP3221_FLClient.py client5 6005 1" to use Mini-Batch GD as the client's optimization method.

It is also ok to start less than 5 clients in the first 30 seconds, and you can start them whenever you want before the server finishes 100 global communication rounds.

After the server's 100 global communication rounds finish, you will see the figures showing the testing accuracies and the training loss of the global model over 100 rounds.
Close the windows showing the figures and the whole program will end.
