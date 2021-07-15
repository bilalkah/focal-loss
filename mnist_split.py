import numpy as np

train_size = 5
test_size = 1000
zero = np.load("mnist-zero.npy")
zero = zero[1:train_size]
print(zero.shape)

dataset = np.append(zero,np.load("mnist-one.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-two.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-three.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-four.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-five.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-six.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-seven.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-eight.npy"),axis=0)
dataset = np.append(dataset,np.load("mnist-nine.npy"),axis=0)
print(dataset.shape)

dataset = dataset.reshape(-1,28,28,1)
print(dataset.shape)
np.save("mnist-dataset", dataset)

target = np.array([0]*(train_size-1))
np.save("mnist-zero-target",target)
target = np.append(target,np.array([1]*np.load("mnist-one.npy").shape[0]),axis=0)
target = np.append(target,np.array([2]*np.load("mnist-two.npy").shape[0]),axis=0)
target = np.append(target,np.array([3]*np.load("mnist-three.npy").shape[0]),axis=0)
target = np.append(target,np.array([4]*np.load("mnist-four.npy").shape[0]),axis=0)
target = np.append(target,np.array([5]*np.load("mnist-five.npy").shape[0]),axis=0)
target = np.append(target,np.array([6]*np.load("mnist-six.npy").shape[0]),axis=0)
target = np.append(target,np.array([7]*np.load("mnist-seven.npy").shape[0]),axis=0)
target = np.append(target,np.array([8]*np.load("mnist-eight.npy").shape[0]),axis=0)
target = np.append(target,np.array([9]*np.load("mnist-nine.npy").shape[0]),axis=0)
print(target.shape)
print(target)

np.save("mnist-target",target)
data = np.load("mnist-dataset.npy")
target = np.load("mnist-target.npy")

print(data.shape)
print(target.shape)

zero = np.load("mnist-zero.npy")
zero = zero[100:(100+test_size)].reshape(-1,28,28,1)
print(zero.shape)
np.save("mnist-zero-test",zero)
target = np.array([0]*zero.shape[0])
np.save("mnist-zero-test-target",target)

data = np.load("mnist-dataset.npy")
print(data.shape)