"""Self-touch model retrained on self-touch episodes + no-object replays of the manipulation gaits (true zeros included).
Same network class as the deployed contrastive FCN, so DexWildMotionController._load_selftouch_model can load it."""
from model.selftouch_contrastive_fcn.selftouch_contrastive_fcn import SelfTouchContrastiveFCN as SelfTouch  # noqa: F401
