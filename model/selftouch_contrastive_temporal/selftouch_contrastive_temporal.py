from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchContrastiveTemporal(SelfTouchSequenceModel):
    """Contrastive encoder using the TSMixer backbone."""

    def __init__(self, param):
        super().__init__(param, backbone="tsmixer", contrastive=True)


SelfTouch = SelfTouchContrastiveTemporal
