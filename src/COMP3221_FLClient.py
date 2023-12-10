# Import necessary packages
import sys
from client_package.client import Client


def main():
    """
    The main function that initializes a client instance and starts the training and testing process.

    This function parses command line arguments for client ID, port number, and optimization method.
    It validates the inputs and starts the register and model receiving processes in separate threads.
    """
    if len(sys.argv) != 4:
        print("Require 4 command line arguments!")
        return
    
    client_id = sys.argv[1]
    port_no = sys.argv[2]
    opt_method = sys.argv[3]
    learning_rate = 0.01
    
    if not client_id or not client_id[-1].isdigit():
        print("Invalid client id! The last character must be a number.")
        return
    
    try:
        port_no = int(port_no)
        if port_no <= 0:
            raise ValueError("Port number must be positive.")
    except ValueError as e:
        print(f"Invalid port number: {e}")
        return
    
    if opt_method not in ["0", "1"]:
        print("Invalid optimization method!")
        return
    
    client = Client(client_id, learning_rate, port_no, opt_method)
    
    if client.register():  # Check if registration is successful
        client.model_receiving()
    else:
        print("Registration failed. Exiting program.")
        return  # Exit if registration fails
    
    
# The starting point of the script. 
if __name__ == "__main__":
    main()