from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchTemporalMixer(SelfTouchSequenceModel):
    """TSMixer backbone for self-touch prediction."""

    def __init__(self, param):
        super().__init__(param, backbone="tsmixer", contrastive=False)


SelfTouchTransformer = SelfTouchTemporalMixer
SelfTouch = SelfTouchTemporalMixer
