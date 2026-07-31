from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchContrastiveFCN(SelfTouchSequenceModel):
    """Contrastive encoder using the TCN-strengthened FCN backbone."""

    def __init__(self, param):
        super().__init__(param, backbone="tcn", contrastive=True)


SelfTouch = SelfTouchContrastiveFCN
