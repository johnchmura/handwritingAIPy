import numpy as np
import pandas as pd
import pygame
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

def init_params():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1. / 784)
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1. / 10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1. / 20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1. / 784)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    print(current_image)
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)

pygame.init()

canvas_size = (28, 28)
pixel_size = 20
button_height = 30
screen_size = (canvas_size[0] * pixel_size, canvas_size[1] * pixel_size + button_height)

black = (0, 0, 0)
white = (255, 255, 255)
grey = (128, 128, 128)
button_color = (100, 100, 100)

screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Draw Digits")

canvas = [[black] * canvas_size[0] for _ in range(canvas_size[1])]

button_width = 60
clear_button_rect = pygame.Rect((screen_size[0] - button_width) // 2 - 100, canvas_size[1] * pixel_size, button_width, button_height)
submit_button_rect = pygame.Rect((screen_size[0] - button_width) // 2 + 100, canvas_size[1] * pixel_size, button_width, button_height)
grey_button_rect = pygame.Rect((screen_size[0] - button_width) // 2 - 50, canvas_size[1] * pixel_size, button_width, button_height)
white_button_rect = pygame.Rect((screen_size[0] - button_width) // 2, canvas_size[1] * pixel_size, button_width, button_height)

running = True
drawing = False
current_color = white  

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button_rect.collidepoint(event.pos):
                canvas = [[black] * canvas_size[0] for _ in range(canvas_size[1])]
            elif submit_button_rect.collidepoint(event.pos):
                flattened_canvas = [255 if canvas[y][x] == white else (128 if canvas[y][x] == grey else 0) for y in range(canvas_size[1]) for x in range(canvas_size[0])]
                
                flattened_canvas = np.array(flattened_canvas).reshape((784, 1)) / 255.0
                prediction = make_predictions(flattened_canvas, W1, b1, W2, b2)
                
                print("Predicted Value:", prediction)
            elif grey_button_rect.collidepoint(event.pos):
                current_color = grey
            elif white_button_rect.collidepoint(event.pos):
                current_color = white
            else:
                x, y = event.pos
                x //= pixel_size
                y //= pixel_size
                if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                    drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            x //= pixel_size
            y //= pixel_size
            if 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                canvas[y][x] = current_color

    for y in range(canvas_size[1]):
        for x in range(canvas_size[0]):
            pygame.draw.rect(screen, canvas[y][x], (x * pixel_size, y * pixel_size, pixel_size, pixel_size))

    pygame.draw.rect(screen, button_color, clear_button_rect)
    font = pygame.font.Font(None, 24)
    text = font.render("Clear", True, (255, 255, 255))
    screen.blit(text, (clear_button_rect.centerx - text.get_width() // 2, clear_button_rect.centery - text.get_height() // 2))

    pygame.draw.rect(screen, button_color, submit_button_rect)
    text = font.render("Submit", True, (255, 255, 255))
    screen.blit(text, (submit_button_rect.centerx - text.get_width() // 2, submit_button_rect.centery - text.get_height() // 2))

    pygame.draw.rect(screen, button_color, grey_button_rect)
    text = font.render("Grey", True, (255, 255, 255))
    screen.blit(text, (grey_button_rect.centerx - text.get_width() // 2, grey_button_rect.centery - text.get_height() // 2))
    
    pygame.draw.rect(screen, button_color, white_button_rect)
    text = font.render("White", True, (255, 255, 255))
    screen.blit(text, (white_button_rect.centerx - text.get_width() // 2, white_button_rect.centery - text.get_height() // 2))

    pygame.display.flip()

pygame.quit()
sys.exit()
