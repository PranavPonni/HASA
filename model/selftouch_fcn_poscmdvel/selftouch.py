from selftouch_fcn_arch import ControlledTemporalSelfTouch


class SelfTouch(ControlledTemporalSelfTouch):
    INPUT_MODALITIES = ('hand_jnt_pos', 'hand_jnt_cmd_pos', 'hand_jnt_vel')


if __name__ == "__main__":
    print("Model for predicting self touch")
