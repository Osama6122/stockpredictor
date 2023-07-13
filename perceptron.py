import numpy as np
import matplotlib.pyplot as plt

def generate_linearly_separable_data(size):
    # Generate random lineraly seperable data for target function X1 = X2
    X1 = [0]*size
    X2 = [0]*size
    for i in range(size):
        X1[i] = np.random.rand()
        X2[i] = np.random.rand()

    Y = [0]*size
    for i in range(size):
        if X2[i] - X1[i] > 0:
            Y[i] = 1
        elif X2[i] - X1[i] < 0:
            Y[i] = -1

    return X1, X2, Y


def perceptron_learning_algorithm(X1, X2, Y,size):
    """Runs PLA starting from initial weights and returns final weigths once learning is complete"""
    w = [0, 0, 0] #Choose Initial weights
    count = i = 0
    #Run PLA
    while i < size:
        sign = np.sign(w[0] + w[1]*X1[i] + w[2]*X2[i])
        #Update weights if data point is misclassified
        if sign != np.sign(Y[i]):
            w[0] = w[0] + Y[i]
            w[1] = w[1] + Y[i] * X1[i]
            w[2] = w[2] + Y[i] * X2[i]
            #Reset the loop because some other data points which were classified correctly previously might be now misclassified due to the update
            i = 0
            count += 1
        else:
            i += 1    
    print(f"Number of iterations for PLA: {count}")

    return w


def plot_graph(X1, X2, Y, W):
    A1, A2, B1, B2 = [], [], [], []

    for i in range(len(X1)):
        if Y[i] == 1:
            A1.append(X1[i])
            B1.append(X2[i])
        else:
            A2.append(X1[i])
            B2.append(X2[i])

    f = np.linspace(0,1,num=20)

    val1 = [0,1]
    val2 = []
    val2.append((-W[0] - (W[1] * val1[0])) / W[2])
    val2.append((-W[0] - (W[1] * val1[1])) / W[2])

    plt.plot(val1, val2, color = 'orange')
    plt.plot(f,f, color = 'green')
    plt.scatter(A1, B1, color = 'red', marker ='s')
    plt.scatter(A2, B2, color = 'blue', marker ='x')
    plt.title("DATA")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()


def main():
    size = 20
    X1, X2, Y  = generate_linearly_separable_data(size)
    W = perceptron_learning_algorithm(X1,X2,Y,size)
    plot_graph(X1, X2, Y, W)

if __name__ == "__main__":
   main()