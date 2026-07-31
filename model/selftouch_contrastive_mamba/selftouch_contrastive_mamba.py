from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchContrastiveMamba(SelfTouchSequenceModel):
    """Contrastive encoder using the selective state-space backbone."""

    def __init__(self, param):
        super().__init__(param, backbone="mamba", contrastive=True)


SelfTouch = SelfTouchContrastiveMamba
