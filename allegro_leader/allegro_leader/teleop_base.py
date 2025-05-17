from abc import ABC, abstractmethod

class AlgoTeleop(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def set_follower_state(self, follower_state):
        ...

    @abstractmethod
    def set_leader_state(self, leader_state):
        ...

    @abstractmethod
    def calc_leader(self):
        ...

    @abstractmethod
    def calc_follower(self):
        ...
