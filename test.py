import numpy as np

def test():
    tran = np.array([0, 2, 3])
    rot = np.array([1, 0, 0, 0])
    return tran, rot

pose = np.concatenate(test(), axis=0)
print(pose)