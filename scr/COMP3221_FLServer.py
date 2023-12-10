# Import necessary packages
import sys
from server_package.server import Server


def main():
    """
    The main function that initializes a server instance and starts the federated learning process.

    This function parses command line arguments for port number, and flag indicating whether clients subsampling is used or not.
    It validates the inputs and starts the training and test process.
    """
    if len(sys.argv) != 3:
        print("Require 3 command line arguments!")
        return
    
    port_no = sys.argv[1]
    sub_client = sys.argv[2]
    
    try:
        port_no = int(port_no)
        if port_no <= 0:
            raise ValueError("Port number must be positive.")
    except ValueError as e:
        print(f"Invalid port number: {e}")
        return
    
    if sub_client not in ["0", "1"]:
        print("Invalid sub-client flag!")
        return
    
    server = Server(port_no, int(sub_client))
    server.run()

# The starting point of the script. 
if __name__ == '__main__':
    main()
