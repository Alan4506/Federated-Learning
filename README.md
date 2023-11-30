# Learning Objectives
![image](https://github.com/Alan4506/Federated-Learning/assets/62124408/c61071e0-eb33-4057-a627-a3517ca23a64)

The task is to implement a simple Federated Learning (FL) system including five clients in total and one server for aggregation in Fig. 1. Each client has its own data used for training its local model and then contributes its local model to the server through the socket in order to build the global model. It is noted that we can simulate the FL system on a single machine by opening different terminals (one each for every client and one for server) on the same machine (use "localhost").

# Federated Learning Algorithm
![image](https://github.com/Alan4506/Federated-Learning/assets/62124408/846f4bce-22fb-435c-9ebd-f0e01f99c9c6)

In this assignment, we implement FedAvg, presented in Alg. 2.1. \( T \) is the total number of global communication rounds between clients and the server. While \( w_t \) is the global model at iteration \( t \), \( w_{t+1}^k \) is the local model of client \( k \) at iteration \( t + 1 \). To obtain the local model, clients can use both Gradient Descent (GD) or Mini-Batch Gradient Descent (Mini-Batch GD) as the optimization methods.
