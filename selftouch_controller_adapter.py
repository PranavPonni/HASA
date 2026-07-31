from model.selftouch_fcn_postrq import controller as _base_controller
from model.selftouch_fcn_postrq.data_loader import CustomDataLoader as SharedCustomDataLoader


class AdaptedRNNController(_base_controller.RNN_controller):
    model_class = None
    data_loader_class = SharedCustomDataLoader

    def train_controller(self, sweep=False):
        if self.model_class is None:
            raise RuntimeError("AdaptedRNNController.model_class must be set")

        old_model_class = _base_controller.SelfTouch
        old_data_loader_class = _base_controller.CustomDataLoader
        _base_controller.SelfTouch = self.model_class
        _base_controller.CustomDataLoader = self.data_loader_class
        try:
            return super().train_controller(sweep=sweep)
        finally:
            _base_controller.SelfTouch = old_model_class
            _base_controller.CustomDataLoader = old_data_loader_class
