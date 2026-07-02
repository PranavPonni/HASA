import os
import sys
import importlib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

sys.modules.pop("selftouch", None)
sys.modules.pop("data_loader", None)

_selftouch_module = importlib.import_module("selftouch")
_data_loader_module = importlib.import_module("data_loader")
_Model = _selftouch_module.SelfTouchTransformer
_Loader = _data_loader_module.CustomDataLoader

_base = importlib.import_module("model.selftouch_transformer.controller")


def _bind_variant():
    sys.modules["selftouch"] = _selftouch_module
    sys.modules["data_loader"] = _data_loader_module
    _base.SelfTouchTransformer = _Model
    _base.CustomDataLoader = _Loader


class RNN_controller(_base.RNN_controller):
    def train_controller(self, sweep=False):
        _bind_variant()
        return super().train_controller(sweep=sweep)

    def test_controller(self):
        _bind_variant()
        return super().test_controller()

    def sweep_controller(self):
        _bind_variant()
        return super().sweep_controller()
