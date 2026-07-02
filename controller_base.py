from abc import ABC, abstractmethod

class AbstractController(ABC):
    @abstractmethod
    def __init__(self,model_param,mode_param,dataset_param,config_param):
        self.model_param=model_param
        self.mode_param=mode_param
        self.dataset_param=dataset_param
        self.config_param=config_param

    @abstractmethod
    def train_controller(self):
        pass
    
    @abstractmethod
    def test_controller(self):
        pass

    @abstractmethod
    def motion_controller(self):
        pass

    def pretrain_controller(self):
        pass


