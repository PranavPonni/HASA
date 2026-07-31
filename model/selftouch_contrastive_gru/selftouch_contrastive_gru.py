from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchContrastiveGRU(SelfTouchSequenceModel):
    """Contrastive encoder using the causal GRU-attention backbone."""

    def __init__(self, param):
        super().__init__(param, backbone="gru_attention", contrastive=True)


SelfTouch = SelfTouchContrastiveGRU
