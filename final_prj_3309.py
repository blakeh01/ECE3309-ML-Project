# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

import tkinter as tk

# TKINTER DEMO APP
class CustomProgressBar(tk.Canvas):

    def __init__(self, master, text, width=100, height=20, color="green"):
        super().__init__(master, width=width, height=height, bg="white", highlightthickness=0)
        self.color = color
        self.rect_width = width
        self.rect_id = None
        self.label = tk.Label(self, text=text, bg="white")

        text_x = self.rect_width / 2
        text_y = self.winfo_reqheight() / 2
        self.label.place_configure(x=text_x, y=text_y, anchor="center")

    def set_value(self, value):
        if self.rect_id:
            self.delete(self.rect_id)
        self.rect_id = self.create_rectangle(0, 0, value * self.rect_width, self.winfo_reqheight(), fill=self.color,
                                             outline="")
        self.tag_lower(self.rect_id)  # Move the rectangle to the bottom to be behind the text


class CanvasApp:

    def __init__(self, root, fpop_func, wh, bh, wo, bo):
        self.root = root
        self.root.title("Canvas App")

        self.fpop_func = fpop_func
        self.wh, self.bh, self.wo, self.bo = wh, bh, wo, bo
        self.guess = None

        # Create frame for canvas
        canvas_frame = tk.Frame(root)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        # Size of the pixel grid
        self.grid_size = 28

        # Create a canvas
        self.canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="white")
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Initialize the pixels array
        self.pixels = [[0] * self.grid_size for _ in range(self.grid_size)]

        # Set up mouse bindings
        self.canvas.bind("<B1-Motion>", self.paint_pixel)

        # Create frame for progress bars
        progress_frame = tk.Frame(root)
        progress_frame.grid(row=0, column=1, padx=10, pady=10)

        # Create custom progress bars for guesses
        self.progress_bars = []

        for digit in range(10):
            progress_bar = CustomProgressBar(progress_frame, digit, width=100, height=20, color="green")
            progress_bar.pack(pady=5)
            self.progress_bars.append(progress_bar)

        # Add a button to plot the array
        self.plot_button = tk.Button(canvas_frame, text="Clear", command=self.clear)
        self.plot_button.pack()

    def clear(self):
        plt.imshow(self.pixels, cmap='gray')  # Use 'gray' colormap for black and white images
        plt.title('Blurred Image')
        plt.show()

        # Clear the canvas
        self.canvas.delete("all")

        # Reset the pixels array
        self.pixels = [[0] * self.grid_size for _ in range(self.grid_size)]

    def paint_pixel(self, event):
        # Get the current mouse position
        x, y = event.x, event.y

        # Map the mouse position to the grid
        grid_x = x // (300 // self.grid_size)
        grid_y = y // (300 // self.grid_size)

        # Paint the grid cell
        self.canvas.create_rectangle(
            grid_x * (300 // self.grid_size),
            grid_y * (300 // self.grid_size),
            (grid_x + 1) * (300 // self.grid_size),
            (grid_y + 1) * (300 // self.grid_size),
            fill="black",
            outline="black"
        )

        # Update the pixels array
        self.pixels[grid_y][grid_x] = 255

        self.guess, _ = self.fpop_func(np.array(self.pixels).flatten(), self.wh, self.bh, self.wo, self.bo)

        for digit, prob in enumerate(self.guess[0]):
            self.progress_bars[digit].set_value(prob)


# FUNCTIONS
# You may add or remove functions according to the need of you code.

def init_weights(Ni, Nh, No):
    # Initializing weights with small random values and setting biases to zero

    # Make sure the matrix size are proper.
    # wh size => [Nh,Ni]
    # wo size => [No,Nh]
    # bh size => [Nh,1]
    # bo size => [No,1]

    wh = np.random.randn(Nh, Ni) * 0.01
    bh = np.zeros((Nh, 1))

    wo = np.random.randn(No, Nh) * 0.01
    bo = np.zeros((No, 1))

    return wh, bh, wo, bo


def forward_pass(X, wh, bh, wo, bo):
    # Compute the output of the hidden layer
    H = sigmoid(np.dot(X, wh.T) + bh.T)

    # Compute the predicted output
    y_hat = softmax(np.dot(H, wo.T) + bo.T)

    return y_hat, H


def backprop(X, y_hat, H, y_true, wo):
    # Output layer error
    delta = (y_hat - y_true) / BZ

    # Hidden layer error
    delta_hidden = delta.dot(wo) * H * (1 - H)

    # dot(delta, wo) * H * (1 - H)

    # Gradients for weights and biases
    g_wo = np.dot(delta.T, H)
    g_bo = np.sum(delta.T)
    g_wh = np.dot(delta_hidden.T, X)
    g_bh = np.sum(delta_hidden.T)

    return g_wo, g_bo, g_wh, g_bh


# Other useful functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sig_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def softmax(z):
    ez = np.exp(z)
    ezs = np.sum(ez, 1, keepdims=True)
    return ez / ezs


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, alpha * z)


def elu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def cost_func(Y, T):
    Er = np.sum(0.5 * (Y - T) ** 2) / Y.shape[0]
    # Er = -T*np.log(Y)-(1-T)*np.log(1-Y)
    # Er = -np.sum(T*np.log(Y))/Y.shape[0]
    return Er


def oneHot(L):
    N = len(L)
    OHT = np.zeros((N, 10))
    for i in range(N):
        OHT[i, L[i, 0]] = 1

    return OHT


def inv_oneHot(OHT):
    L = np.size(OHT, 0)
    YY = np.zeros((L, 1))
    for i in range(L):
        YY[i] = np.argmax(OHT[i, :])
    return YY


def save_neural_network_parameters(wh, bh, wo, bo, filename='neural_params_trained.pkl'):
    parameters = {'wh': wh, 'bh': bh, 'wo': wo, 'bo': bo}

    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)
        print("Saved neural network parameters to ", filename, "!")


def load_neural_network_parameters(filename='neural_params_trained.pkl'):
    with open(filename, 'rb') as file:
        parameters = pickle.load(file)
        print("Loaded neural network parameters from ", filename, "!")

    return parameters['wh'], parameters['bh'], parameters['wo'], parameters['bo']


################################ Load Dataset ################################

# Replace with your directory
hm_dir = 'C:/Users/bmhav/Desktop/ece3309_final/data/'

# Replace with your filename (CSV)
fname = 'mnist_train.csv'
fname_test = 'mnist_test.csv'

# Load data and prepare Input/Output matrices
full_file = os.path.join(hm_dir, fname)
data = np.array(pd.read_csv(full_file))

full_file_test = os.path.join(hm_dir, fname_test)
data_test = np.array(pd.read_csv(full_file_test))

LBL = np.reshape(data[:, 0], (60000, 1))  # True Labels
MAT = data[:, 1:]  # Image data

LBL_test = np.reshape(data_test[:, 0], (10000, 1))  # True Labels
MAT_test = data_test[:, 1:]  # Image data

Nf = np.size(MAT, 1)  # Number of input features

# Preparing Training and Testing datasets

X = MAT  # Training data matrix
T = oneHot(LBL)  # Training Labels
N = X.shape[0]  # Number of training samples

X_test = MAT_test  # Testing data matrix
T_test = oneHot(LBL_test)  # Testing labels
N_test = X_test.shape[0]  # Number of testing samples

if input("Load weights/biases from file? (y/n)").lower() == 'y':
    wh, bh, wo, bo = load_neural_network_parameters()
else:
    Ni = Nf  # Number of input feature (28x28: one per pixel)
    No = 10  # Number of output classes (10 digits)
    Nh = int(input("Enter number of hidden nodes: "))  # Number of hidden layer nodes

    # YOUR CODE to initialize weights and bias

    # Uncomment and modify code below.
    wh, bh, wo, bo = init_weights(Ni, Nh, No)
    # DO NOT CHANGE VARIABLE NAMES

    # Training the network
    # You may run this module multiple times to keep on training and fine tuning your network.
    # Usually 3 (hyper)parameters are modified with each fine-tuning run
    # You may play around with different values for the 3 variables below.

    BZ = 32  # Minibatch size
    # (Larger number will result in slower training and smaller number will result in poor training)

    EPH = int(input("Enter desired epochs: "))  # Epochs (How many times to train over entire dataset.)

    lr = 0.1  # Learning rate. (Start with 0.01 and then tweak it to make perfect.)
    # Too large learning rates may result in divering error and will blow up to a large number
    # Too small learning rate may result in very very slow training.
    # You may reduce this number gradually as the network trains to make accuracy better and better.

    MaxItr = np.int64(np.floor(N / BZ))
    Err = np.zeros((MaxItr * EPH, 1))

    cnt = 0
    for ep in range(EPH):

        # Select minibatch
        i_rnd = np.random.permutation(N)
        XX = X[i_rnd, :]
        TT = T[i_rnd, :]

        lr -= 0.001
        if lr <= 0.02:
            lr = 0.02

        for itrr in range(MaxItr):
            s = itrr * BZ
            e = (itrr + 1) * BZ

            # Your function or code to do forward propagation
            # Forward pass requires minibatch, weights and bias.
            # It should output prediction and hidden layer output
            # Do not change the variable name otherwise the code might not work
            # Uncomment the line below and put your own code or your function name

            y_hat, H = forward_pass(XX[s:e, :], wh, bh, wo, bo)

            # Your function or code to do backpropagation
            # Backpropagation should output gradients corresponding to weights and bias
            # The input and output variable names are for reference.
            # Do not change the variable names otherwise the code might not work.
            # Uncomment the line below and put your own code or your function name
            g_wo, g_bo, g_wh, g_bh = backprop(XX[s:e, :], y_hat, H, TT[s:e], wo)

            wo = wo - lr * g_wo
            wh = wh - lr * g_wh
            bo = bo - lr * g_bo
            bh = bh - lr * g_bh

            Err[cnt] = cost_func(y_hat, TT[s:e, :])
            cnt = cnt + 1

        print(np.mean(Err[cnt - MaxItr:cnt - 1]))  # This will print error for each epoch

# This section will show the performance on unseen data.
# If you are not using functions, please replace the function below with your implementation.
OHT, Hpred = forward_pass(X_test, wh, bh, wo, bo)

Ypred = inv_oneHot(OHT)
TT = inv_oneHot(T_test)
acc = np.sum(np.round(Ypred) == TT)

Tst_acc = acc / N_test * 100
print('Testing Accuracy = ', Tst_acc)

save_neural_network_parameters(wh, bh, wo, bo)

df = pd.DataFrame (wh)
df.to_excel('trained_wh.xlsx', index=False,header=False)
df = pd.DataFrame (wo)
df.to_excel('trained_wo.xlsx', index=False,header=False)
df = pd.DataFrame (bh)
df.to_excel('trained_bh.xlsx', index=False,header=False)
df = pd.DataFrame (bo)
df.to_excel('trained_bo.xlsx', index=False,header=False)

## Display testing results

plt.figure(figsize=[10, 6])
for i in range(32):
    idx = np.random.randint(N_test)
    I = np.reshape(X_test[idx, :], (28, 28))
    plt.subplot(4, 8, i + 1)
    plt.imshow(I)
    plt.title(str(np.round(Ypred[idx])))
    plt.gray()

plt.show()

root = tk.Tk()
app = CanvasApp(root, forward_pass, wh, bh, wo, bo)
root.mainloop()
