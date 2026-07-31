from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchGRUAttention(SelfTouchSequenceModel):
    """Causal GRU backbone with attention refinement for self-touch prediction."""

    def __init__(self, param):
        super().__init__(param, backbone="gru_attention", contrastive=False)


SelfTouchTransformer = SelfTouchGRUAttention
SelfTouch = SelfTouchGRUAttention
