from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchTransformer(SelfTouchSequenceModel):
    """PatchTST-style transformer backbone for self-touch prediction."""

    def __init__(self, param):
        super().__init__(param, backbone="patchtst", contrastive=False)


SelfTouch = SelfTouchTransformer
