from model.selftouch_contrastive.data_loader import SelfTouchContrastiveDataset
try:
    SelfTouchContrastiveDataset(
        data_dir='/home/handling04/Documents/HASA/data_server/selftouch_all',
        sequence_length=16,
        stride=4,
        combinations=['thumb-index', 'thumb-middle', 'index-middle'],
    )
    print("PASSED: no error raised")
except RuntimeError as e:
    lines = str(e).splitlines()
    for l in lines[:25]:
        print(l)
