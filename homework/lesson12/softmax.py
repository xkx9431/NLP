import numpy as np
import random

vector = np.random.random((1, 100))
hidden = np.random.random((100, 10000))
def softmax(logits):
    logits = logits - np.max(logits) # normalization
    with_exp = np.exp(logits)
    return with_exp / np.sum(with_exp)
result = np.dot(vector, hidden)


print(softmax(result)[0])