import numpy as np

def randomization(n, m=1):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    # Your code here
    if n < 1:
        print("n must be an integer greater than 0")
    else:
        return np.random.rand(n, m)


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    # Your code here
    if h < 1 or w < 1:
        print("h and w must be integers greater than equal to 1")
    else:
        A = np.random.rand(h, w)
        B = np.random.rand(h, w)
        s = A + B
        return A, B, s


def norm(A, B):
    return np.linalg.norm(A + B)


def neural_network(inputs, weights):
    return np.tanh(np.dot(np.transpose(weights), inputs))


def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    # Your code here
    if x <= y:
        return x * y
    else:
        return x / y


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    # Your code here
    vec = np.vectorize(scalar_function)
    return vec(x, y)


def lmbda(i):
    return lambda x: x + i

def get_sum_metrics(predictions, metrics=[]):
    #print(len(metrics))
    for i in range(3):
        metrics.append(lmbda(i))

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)
    metrics.clear()
    return sum_metrics


for j in range(6):
    print(get_sum_metrics(j))

metrics = []
for i in range(3):
       metrics.append(lambda x: x + i)
print(metrics)
print(vector_function([21,3,4],[4,5,3]))
print(scalar_function(2,3))
print(np.tanh(np.dot(np.transpose(np.random.rand(2,1)),np.random.rand(2,1))))

print(neural_network([[0.045919],[0.75134221]], [[0.28864077],[0.81478003]]))
print(norm([2,3,3],[2,5,4]))
print(randomization(5))
print(operations(3,2))

def perceptron_hw():
    theta1 = -3
    theta2 = 3
    theta0 = -1.5

    y = [1, 1, -1, -1, -1]
    x1 = [-4, -2, -1, 2, 1]
    x2 = [ 2, 1, -1, 2, -2]
    r =10
    for j in range(0,r):
        i = j % len(x1)
        print(j, "theta:", theta1, theta2, theta0, "X:", x1[i], x2[i], "y:", y[i])
        if (x1[i] * theta1 + x2[i] * theta2+theta0) == 0:
            theta1 = theta1 + y[i] * x1[i]
            theta2 = theta2 + y[i] * x2[i]
            theta0 = theta0 + y[i]

        elif (x1[i] * theta1 + x2[i] * theta2 + theta0) * y[i] < 0:
                theta1 = theta1 + y[i] * x1[i]
                theta2 = theta2 + y[i] * x2[i]
                theta0 = theta0 + y[i]
                #print(theta1, theta2, x1[i], x2[i], y[i])
                print(j, "theta:", theta1, theta2, theta0, "X:", x1[i], x2[i], "y:", y[i])


perceptron_hw()

