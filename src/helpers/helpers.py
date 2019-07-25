import numpy as np
from matplotlib import pyplot as plt

# Usage examples in notebooks/classifiers/SVM.ipynb

def draw_decision_boundary_2d(data,classifier,real_labels):
    h = 0.2
    x_min, x_max = data[:,0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:,1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

    # create decision boundary plot
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z, alpha=0.4)
    plt.scatter(data[:,0],data[:,1],c=real_labels)
    
def shuffle(samples, labels):
    a = np.matrix(labels).T
    b = np.matrix(samples)
    merged = np.concatenate((a,b), axis=1)
    np.random.shuffle(merged)
    shuffled_samples = merged[:,1:]
    shuffled_labels = merged[:,0]
    return shuffled_samples, shuffled_labels

def training_test_split(samples, labels, training_size=0.8, shuffle=True):
    split_index = int(labels.size * training_size)
    training_samples = samples[:split_index]
    training_labels = labels[:split_index]
    test_samples = samples[split_index:]
    test_labels = labels[split_index:]
    return training_samples, training_labels, test_samples, test_labels