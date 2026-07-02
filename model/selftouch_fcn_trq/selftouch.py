from selftouch_fcn_arch import ControlledTemporalSelfTouch


class SelfTouch(ControlledTemporalSelfTouch):
    INPUT_MODALITIES = ('hand_jnt_trq',)


if __name__ == "__main__":
    print("Model for predicting self touch")
