import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split

# generates x and y numpy arrays for
# y = a*x + b + a * noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_linear(a, b, noise, filename, size=100):
    print('Generating random data y = a*x + b')
    x = 2 * np.random.rand(size, 1) - 1
    y = a * x + b + noise * a * (np.random.rand(size, 1) - 0.5)
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')
    return (x, y)


# thats an example of linear regression using polyfit
def linear_regression_numpy(filename):
    # now let's read it back
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    # split to initial arrays
    x, y = np.hsplit(data, 2)
    # printing shapes is useful for debugging
    print(np.shape(x))
    print(np.shape(y))
    # our model
    time_start = time()
    model = np.polyfit(np.transpose(x)[0], np.transpose(y)[0], 1)
    time_end = time()
    print(f"polyfit in {time_end - time_start} seconds")
    # our hypothesis for give x
    h = model[0] * x + model[1]

    # and check if it's ok
    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r", label='model')
    plt.legend()
    plt.show()
    return (model)


def linear_regression_exact(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)

    X = np.hstack([x, np.ones((x.shape[0], 1))])

    time_start = time()

    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    time_end = time()
    print(f"Inversion method in {time_end - time_start} seconds")

    h = X @ beta

    plt.title("Linear regression task")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='experiment')
    plt.plot(x, h, "r-", label='model')
    plt.legend()
    plt.show()

    return beta


def check(model, ground_truth):
    if len(model) != len(ground_truth):
        print("Model is inconsistent")
        return False
    else:
        r = np.dot(model - ground_truth, model - ground_truth) / (np.dot(ground_truth, ground_truth))
        print(r)
        if r < 0.0001:
            return True
        else:
            return False


# Ex1: make the same with polynoms

# generates x and y numpy arrays for
# y = a_n*X^n + ... + a2*x^2 + a1*x + a0 + noise
# in range -1 .. 1
# with random noise of given amplitude (noise)
# vizualizes it and unloads to csv
def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    print(np.shape(x))
    data = np.hstack((x, y))
    np.savetxt(filename, data, delimiter=',')


def create_polynomial_features(x, degree):

    X_poly = np.ones((x.shape[0], 1))
    for i in range(1, degree + 1):
        X_poly = np.hstack((X_poly, np.power(x, i)))
    return X_poly


def polynomial_regression(filename, degree):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    x, y = np.hsplit(data, 2)

    X_poly = create_polynomial_features(x, degree)

    beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    x_range = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
    X_poly_range = create_polynomial_features(x_range, degree)
    y_pred = X_poly_range @ beta

    plt.title("Polynomial regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y, "b.", label='Experiment data')
    plt.plot(x_range, y_pred, "r-", label=f'Polynomial degree {degree}')
    plt.legend()
    plt.show()

    return beta


# Ex.2 gradient descent for linear regression without regularization

# find minimum of function J(theta) using gradient descent
# alpha - speed of descend
# theta - vector of arguments, we're looking for the optimal ones (shape is 1 х N)
# J(theta) function which is being minimizing over theta (shape is 1 x 1 - scalar)
# dJ(theta) - gradient, i.e. partial derivatives of J over theta - dJ/dtheta_i (shape is 1 x N - the same as theta)
# x and y are both vectors

def gradient_descent_step(dJ, theta, alpha):
    theta = theta - alpha * dJ
    return theta


# get gradient over all xy dataset - gradient descent
def get_dJ(x, y, theta):
    m = len(y)  # Number of examples
    predictions = x.dot(theta)
    error = predictions - y
    dJ = (1 / m) * x.T.dot(error)
    return dJ


# get gradient over all minibatch of size M of xy dataset - minibatch gradient descent
def get_dJ_minibatch(x, y, theta, M):
    indices = np.random.choice(range(len(y)), M, replace=False)
    x_batch = x[indices]
    y_batch = y[indices]
    return get_dJ(x_batch, y_batch, theta)


# get gradient over all minibatch of single sample from xy dataset - stochastic gradient descent
def get_dJ_sgd(x, y, theta):
    i = np.random.randint(0, len(y))
    x_i = x[i:i + 1]
    y_i = y[i:i + 1]
    return get_dJ(x_i, y_i, theta)

def read_data_from_file(filename):
    # Загрузка данных из файла
    data = np.loadtxt(filename, delimiter=',')
    # Первый столбец - y, остальные - x
    y = data[:, 0]
    x = data[:, 1:]
    # Добавление столбца единиц к x для учета свободного члена (интерцепта)
    x = np.hstack([np.ones((x.shape[0], 1)), x])
    return x, y
def minimize(filename, L, alpha, get_gradient_func):
    x, y = read_data_from_file(filename)
    n = x.shape[1]  # Количество признаков
    theta_initial = np.zeros(n)
    theta = np.copy(theta_initial)
    J_history = []

    for i in range(L):
        dJ = get_gradient_func(x, y, theta)
        theta = gradient_descent_step(dJ, theta, alpha)
        J = np.mean((x.dot(theta) - y) ** 2) / 2
        J_history.append(J)

    # Строим график J(theta) по итерациям
    plt.plot(J_history, label=f'Alpha {alpha}')
    plt.xlabel('Итерация')
    plt.ylabel('J(theta)')
    plt.title('Функция стоимости по итерациям')
    plt.legend()
    plt.show()

    return theta


def generate_and_split_data(degree, sizes):
    results = []
    for size in sizes:
        # Генерация данных
        a = np.random.randn(degree + 1)  # Случайные коэффициенты для полинома
        noise = 0.1
        filename = f"poly_data_{size}.csv"
        generate_poly(a, degree, noise, filename, size)

        # Чтение и разбиение данных
        data = np.loadtxt(filename, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        results.append((X_train, X_test, X_valid, y_train, y_test, y_valid))
    return results
sizes = range(10, 101, 10)
data_splits = generate_and_split_data(3, sizes)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost


def polynomial_features(X, degree):
    X_poly = X
    for i in range(2, degree + 1):
        X_poly = np.hstack((X_poly, np.power(X[:, 0:1], i)))
    return X_poly


def train_and_evaluate(X_train, X_test, y_train, y_test, degrees, L, alpha):
    J_train = []
    J_test = []

    for degree in degrees:
        # Генерация полиномиальных признаков
        X_train_poly = polynomial_features(X_train, degree)
        X_test_poly = polynomial_features(X_test, degree)

        # Добавление столбца единиц к X
        X_train_poly = np.hstack([np.ones((X_train_poly.shape[0], 1)), X_train_poly])
        X_test_poly = np.hstack([np.ones((X_test_poly.shape[0], 1)), X_test_poly])

        # Обучение модели
        theta_initial = np.zeros(X_train_poly.shape[1])
        theta_optimized = minimize(theta_initial, X_train_poly, y_train, L, alpha, get_dJ)

        # Вычисление стоимости
        cost_train = compute_cost(X_train_poly, y_train, theta_optimized)
        cost_test = compute_cost(X_test_poly, y_test, theta_optimized)

        J_train.append(cost_train)
        J_test.append(cost_test)

    return J_train, J_test

def minimize_with_prepared_data(X_train, y_train, X_test, y_test, L, alpha, get_gradient_func):
    n = X_train.shape[1]  # Количество признаков
    theta_initial = np.zeros(n)
    J_history_train = []
    J_history_test = []

    for i in range(L):
        dJ = get_gradient_func(X_train, y_train, theta_initial)
        theta_initial = gradient_descent_step(dJ, theta_initial, alpha)
        J_train = np.mean((X_train.dot(theta_initial) - y_train) ** 2) / 2
        J_test = np.mean((X_test.dot(theta_initial) - y_test) ** 2) / 2
        J_history_train.append(J_train)
        J_history_test.append(J_test)

    # Строим графики J(theta) по итерациям
    plt.plot(J_history_train, label='Train Cost', alpha=0.75)
    plt.plot(J_history_test, label='Test Cost', alpha=0.75)
    plt.xlabel('Итерация')
    plt.ylabel('J(theta)')
    plt.title(f'Функция стоимости по итерациям, Alpha {alpha}')
    plt.legend()
    plt.show()

    return theta_initial
def train_and_evaluate_with_minimize(X_train, X_test, y_train, y_test, degrees, L, alpha):
    J_train = []
    J_test = []

    for degree in degrees:
        # Добавляем полиномиальные признаки
        X_train_poly = polynomial_features(X_train, degree)
        X_test_poly = polynomial_features(X_test, degree)

        # Обучение модели и минимизация функции стоимости
        theta_optimized = minimize_with_prepared_data(X_train_poly, y_train, X_test_poly, y_test, L, alpha, get_dJ)

        # Вычисляем стоимость для оптимизированного theta
        J_train.append(compute_cost(X_train_poly, y_train, theta_optimized))
        J_test.append(compute_cost(X_test_poly, y_test, theta_optimized))

    return J_train, J_test


def experiment(sizes, data_splits, degrees, L, alpha):
    for size, splits in zip(sizes, data_splits):
        X_train, X_test, X_valid, y_train, y_test, y_valid = splits
        J_train, J_test = train_and_evaluate_with_minimize(X_train, X_test, y_train, y_test, degrees, L, alpha)

        # Визуализация min(J_train), min(J_test) в зависимости от степени полинома
        plt.plot(degrees, J_train, label='Train Cost', marker='o')
        plt.plot(degrees, J_test, label='Test Cost', marker='x')
        plt.title(f'Model Evaluation for size={size}')
        plt.xlabel('Degree of Polynomial')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    generate_linear(1, -3, 1, 'linear.csv', 100)
    #model = linear_regression_numpy("linear.csv")
    #print(f"Is model correct?\n{check(model, np.array([1, -3]))}")
    # ex1 . - exact solution
    model_exact = linear_regression_exact("linear.csv")

    # ex1. polynomial with numpy
    generate_poly([1, 2, 3], 2, 0.5, 'polynomial.csv')
    polynomial_regression("polynomial.csv", 2)

   # ex2.
    filename = 'linear.csv'
    alpha = 0.01
    L = 1000
    minimize(filename, L, alpha, get_dJ)
    # ex3. polinomial regression
    sizes = range(10, 101, 10)
    data_splits = generate_and_split_data(3, sizes)
    degrees =[2,3,4]
    experiment(sizes, data_splits, degrees, L, alpha)

    # ex3* the same with regularization
