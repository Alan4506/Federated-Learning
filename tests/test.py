import unittest
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from server_package.server import Server
from server_package.user import User

class TestServer(unittest.TestCase):
    def setUp(self):
        self.server = Server(6000, 1)
        self.user1 = User(100, 6001, "111")
        self.user2 = User(200, 6002, "222")
        self.user3 = User(300, 6003, "333")
        self.user4 = User(400, 6004, "444")

    def test_check_user_existence(self):
        self.server.users = [self.user1, self.user2]
        self.assertTrue(self.server.check_user_existence("111"))  
        self.assertTrue(self.server.check_user_existence("222"))  
        self.assertFalse(self.server.check_user_existence("333"))  

    def test_drop_dead_clients(self):
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
        

if __name__ == '__main__':
    unittest.main()
