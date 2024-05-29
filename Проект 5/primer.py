import numpy as np
import random
from dataset_train import dataset


INPUT = 36
HIDE1 = 30
HIDE2 = 20
HIDE3 = 20  
HIDE4 = 10 
OUT = 5


test_data = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]])




# Вагова матриця від входу до першого прихованого шару
W1 = np.random.randn(INPUT, HIDE1)
# Вагова матриця від першого прихованого до другого прихованого шару
W2 = np.random.randn(HIDE1, HIDE2)
# Вагова матриця від другого прихованого до третього прихованого шару
W3 = np.random.randn(HIDE2, HIDE3)
# Вагова матриця від третього прихованого до четвертого прихованого шару
W4 = np.random.randn(HIDE3, HIDE4)
# Вагова матриця від четвертого прихованого до виходу
W5 = np.random.randn(HIDE4, OUT)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def relu(t):
    return np.maximum(t, 0)

def relu_deriv(t):
    return (t >= 0).astype(float)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def sparse_cross_entropy(z, y):
    return -np.log(z[y])

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

EPOCH = 10000
learning_rate = 0.01

for ep in range(EPOCH):
    for i in range(len(dataset)):
        dataset = [(x.flatten(), y) for x, y in dataset]
        x, y = dataset[i]
        
        # Forward pass
        S1 = x @ W1
        h1 = sigmoid(S1)
        S2 = h1 @ W2
        h2 = sigmoid(S2)
        S3 = h2 @ W3
        h3 = sigmoid(S3)
        S4 = h3 @ W4
        h4 = sigmoid(S4)
        S5 = h4 @ W5
        z = softmax(S5)
        err = sparse_cross_entropy(z, y)

        # Backpropagation
        y_full = to_full(y, OUT)
        
        dS5 = z - y_full 
        dW5 = np.outer(h4, dS5)
        
        dh4 = dS5 @ W5.T
        dS4 = dh4 * sigmoid_derivative(S4) 
        dW4 = np.outer(h3, dS4)

        dh3 = dS4 @ W4.T
        dS3 = dh3 * sigmoid_derivative(S3) 
        dW3 = np.outer(h2, dS3)

        dh2 = dS3 @ W3.T
        dS2 = dh2 * sigmoid_derivative(S2) 
        dW2 = np.outer(h1, dS2)

        dh1 = dS2 @ W2.T
        dS1 = dh1 * sigmoid_derivative(S1) 
        dW1 = np.outer(x, dS1)

        # Update 
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2
        W3 -= learning_rate * dW3
        W4 -= learning_rate * dW4
        W5 -= learning_rate * dW5
        
print('Навчання')
for i in range(len(dataset)):
    x, y = dataset[i]
    S1 = x @ W1
    h1 = sigmoid(S1)
    S2 = h1 @ W2
    h2 = sigmoid(S2)
    S3 = h2 @ W3
    h3 = sigmoid(S3)
    S4 = h3 @ W4
    h4 = sigmoid(S4)
    S5 = h4 @ W5
    print(f"Потрібно: {y}, Отримано: {np.argmax(S5)}")

def predict(x):
    x = x.flatten()
    S1 = np.matmul(x, W1)
    h1 = sigmoid(S1)
    S2 = np.matmul(h1, W2)
    h2 = sigmoid(S2)
    S3 = np.matmul(h2, W3)
    h3 = sigmoid(S3)
    S4 = np.matmul(h3, W4)
    h4 = sigmoid(S4)
    S5 = np.matmul(h4, W5)
    z = softmax(S5)
    return z

probs = predict(test_data)
print(probs)
pred_class = np.argmax(probs)
class_name = ['KVADRAT', 'KRUG', 'TRUKYTNUK', 'ROMB', 'ELIPS']
print('Результат: ', class_name[pred_class])
