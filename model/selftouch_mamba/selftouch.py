from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchMamba(SelfTouchSequenceModel):
    """Selective state-space backbone for self-touch prediction."""

    def __init__(self, param):
        super().__init__(param, backbone="mamba", contrastive=False)


SelfTouch = SelfTouchMamba
