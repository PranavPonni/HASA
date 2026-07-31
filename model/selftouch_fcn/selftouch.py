from selftouch_sequence_models import SelfTouchSequenceModel


class SelfTouch(SelfTouchSequenceModel):
    """TCN-strengthened FCN baseline for four-finger self-touch prediction."""

    def __init__(self, param):
        super().__init__(param, backbone="tcn", contrastive=False)


if __name__ == "__main__":
    print("Model for predicting self touch")
