from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouchContrastiveTransformer(SelfTouchSequenceModel):
    """Contrastive encoder using the PatchTST-style transformer backbone."""

    def __init__(self, param):
        super().__init__(param, backbone="patchtst", contrastive=True)


SelfTouch = SelfTouchContrastiveTransformer
