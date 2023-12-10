# Import necessary packages
import unittest
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from server_package.server import Server
from server_package.user import User

class TestServer(unittest.TestCase):
    """
    Unit tests for the Server class.
    """
    
    def setUp(self):
        """
        Set up the test environment before each test.

        This method initializes the server and four users with specified parameters.
        """
        self.server = Server(6000, 1)
        self.user1 = User(100, 6001, "111")
        self.user2 = User(200, 6002, "222")
        self.user3 = User(300, 6003, "333")
        self.user4 = User(400, 6004, "444")

    def test_check_user_existence(self):
        """
        Test the check_user_existence method of the Server class.

        This method verifies if the check_user_existence method correctly identifies
        whether users are existing in the server's user list or not.
        """
        self.server.users = [self.user1, self.user2]
        self.assertTrue(self.server.check_user_existence("111"))  
        self.assertTrue(self.server.check_user_existence("222"))  
        self.assertFalse(self.server.check_user_existence("333"))  

    def test_drop_dead_clients(self):
        """
        Test the drop_dead_clients method of the Server class.

        This method assesses if the drop_dead_clients method accurately removes dead users.
        """
        self.server.users = [self.user3, self.user4]
        self.server.current_iteration = 50
        self.user3.current_iteration = 50
        self.user4.current_iteration = 49
        self.iteration_time = time.time() - 100
        self.server.drop_dead_clients()
        self.assertEqual(len(self.server.users), 1)
        self.assertTrue(self.user3 in self.server.users)  
        self.assertTrue(self.user4 not in self.server.users)  

    def test_check_all_received(self):
        """
        Test the check_all_received method of the Server class.

        This method checks if the check_all_received method accurately determines
        whether all users' packets have been received by the server.
        """
        self.server.users = [self.user1, self.user3]
        self.server.current_iteration = 50
        self.user1.current_iteration = 49
        self.user3.current_iteration = 49
        self.assertFalse(self.server.check_all_received())  

        self.server.current_iteration = 50
        self.user1.current_iteration = 50
        self.user3.current_iteration = 49
        self.assertFalse(self.server.check_all_received())  

        self.server.current_iteration = 50
        self.user1.current_iteration = 50
        self.user3.current_iteration = 50
        self.assertTrue(self.server.check_all_received())  

    def test_evaluate(self):
        """
        Test the evaluate method of the Server class.

        This method checks if the evaluate method correctly computes and returns
        the average accuracy and loss of all users on the server.
        """
        self.server.users = []
        self.assertEqual(self.server.evaluate(), (0, 0))

        self.server.users = [self.user1, self.user2, self.user3]
        self.user1.accuracy = 0.3
        self.user2.accuracy = 0.4
        self.user3.accuracy = 0.2
        self.user1.loss = 0.01
        self.user2.loss = 0.02
        self.user3.loss = 0.03
        self.assertEqual(self.server.evaluate(), (0.3, 0.02))
        

# The starting point of the test script. 
if __name__ == '__main__':
    unittest.main()
