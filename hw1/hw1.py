import numpy as np
import matplotlib.pyplot as plt
from math import exp, cos

train_data = np.genfromtxt("data/earth_temperature_sampled_train.csv", delimiter = ',')
year_train = train_data[:, 0] / 1000
temp_train = train_data[:, 1]
test_data = np.genfromtxt("data/earth_temperature_sampled_test.csv", delimiter = ',')
year_test = test_data[:, 0] / 1000
temp_test = test_data[:, 1]

from T1_P1_TestCases import test_p1
from T1_P2_TestCases import test_p2


# PROBLEM 1

def kernel(x, x_prime, tau):
    return exp(-(x-x_prime) ** 2 / tau)


def kernel_regressor(x_new, tau, x_train, y_train):
    """
    Run f_tau(x) with parameter tau on every entry of x_array.

    :param x_new: a numpy array of x_values on which to do prediction. Shape is (n,)
    :param float tau: lengthscale parameter
    :param x_train: the x coordinates of the training set
    :param y_train: the y coordinates of the training set
    :return: if x_array = [x_1, x_2, ...], then return [f(x_1), f(x_2), ...]
             where f is calculated wrt to the training data and tau
    """
    # DONE: implement this
    
    pred = []

    for new_x in x_new:
        num, den = 0, 0
        for train_ind in range(len(x_train)):
            num += kernel(x_train[train_ind], new_x, tau) * y_train[train_ind]
            den += kernel(x_train[train_ind], new_x, tau)
        pred.append(num / den)

    return pred



test_p1(kernel_regressor)

# plot functions
x_array = np.arange(400, 800 + 1, 1)
for tau in [1, 50, 2500]:
    plt.plot(x_array, kernel_regressor(x_array, tau, year_train, temp_train), label = f"$\\tau = {tau}$")
plt.scatter(year_train, temp_train, label = "training data", color = "red")
plt.legend()
plt.xticks(np.arange(400, 800 + 100, 100))
plt.ylabel("Temperature")
plt.xlabel("Year BCE (in thousands)")
plt.ylim([-5,7])

plt.gca().invert_xaxis()

# figure should be in your directory now, with name p1.2.png
plt.savefig("images/p1.2.png", bbox_inches = "tight")
plt.show()

def model_mse(predictions, true):
    """
    Calculate the MSE for the given model predictions, with respect to the true values

    :param predictions: predictions given by the model
    :param true: corresponding true values
    :return: the mean squared error
    """
    # DONE: implement this
    mse = 0

    for pred, val in zip(predictions, true):
        mse += (val - pred) ** 2

    return mse / len(predictions)

for tau in [1, 50, 2500]:
    print(f"tau = {tau}: loss = {model_mse(kernel_regressor(year_test, tau, year_train, temp_train), temp_test)}")





# PROBLEM 2

from statistics import mean

def predict_knn(x_new, k, x_train, y_train):
    """
    Returns predictions for the values in x_test, using KNN predictor with the specified k.

    :param x_new: a numpy array of x_values on which to do prediction. Shape is (n,)
    :param k: number of nearest neighbors to consider
    :param x_train: x coordinates of training dataset
    :param y_train: y coordinates of training dataset

    :return: if x_array = [x_1, x_2, ...], then return [f(x_1), f(x_2), ...]
             where f is the kNN with specified parameters and training set
    """
    
    if k == 0:
      return x_new
    
    pred = []

    for new_x in x_new:
       dist = np.zeros(len(x_train))
       for i in range(len(x_train)):
          dist[i] = kernel(x_train[i], new_x, 2500)

       ind = dist.argsort()[-k:]

       near_vals = []
       for j in range(k):
          near_vals.append(y_train[ind[j]])

       pred.append(mean(near_vals))

    return pred

test_p2(predict_knn)

# plot functions
N = year_train.shape[0]
x_array = np.arange(400, 800, 1)
plt.plot(x_array, predict_knn(x_array, 1, year_train, temp_train), label = "$k = 1$")
plt.plot(x_array, predict_knn(x_array, 3, year_train, temp_train), label = "$k = 3$")
plt.plot(x_array, predict_knn(x_array, N - 1, year_train, temp_train), label = "$k = N - 1$")
plt.scatter(year_train, temp_train, label = "training data", color = "red")
plt.ylabel("Temperature")
plt.xlabel("Year BCE (in thousands)")

plt.legend()
plt.xticks(np.arange(400, 900, 100))
plt.ylim([-5,7])

plt.gca().invert_xaxis()
# figure should be in your directory now, with name p2.1.png
plt.savefig("images/p2.1.png", bbox_inches = "tight")
plt.show()

# choose your value of k and calculate the loss
for k in [1, 3, 55]:
    print(model_mse(predict_knn(year_test, k, year_train, temp_train), temp_test))






# Problem 3

## don't change anything here
def f_scale(X, part = "a"):
  if part == "a":
    X = X/181 # 181000
  elif part == "b":
    X = X/4e2 # 4e5
  elif part == "c":
    X = X/1.81 # 1810    
  elif part == "d":
    X = X/.181 # 181
  return X
###

# DONE: Complete this `make_basis` function according to the above
# specifications. The function should return the array `phi(X)`
def make_basis(X,part='a'):
  """
  Args:
    X: input of years (or any variable you want to turn into the appropriate basis) as
      ndarray with length `N`.
    part: one of `a`, `b`, `c`, `d` depending on the basis function.

  Returns:
    ndarray `phi(X)` of shape `(N,D)`. For each part the shapes of your
    training data `make_basis(years_train)` should be
      (a) 57x10, (b) 57x10, (c) 57x10, (d) 57x50.
  """
  
  phi_X = []
  ### DO NOT CHANGE THIS SECTION 
  ### it is to prevent numerical instability from taking the exponents of
  ### the years, as well as break symmetry when dealing with a Fourier basis.
  X = f_scale(X, part)
  ### end section

  if part == 'a':
    # DONE: Implement this
    for scaledX in X:
        row = [1]
        for i in range(1,10):
            row.append(scaledX ** i)
        phi_X.append(row)
  elif part=='b':
    # DONE
    for scaledX in X:
        row = [1]
        for i in range(1,10):
            mu = (i + 7) / 8
            row.append(exp(-(((scaledX - mu) ** 2) / 5)))
        phi_X.append(row)
  elif part=='c':
    # DONE
    for scaledX in X:
        row = [1]
        for i in range(1,10):
            row.append(cos(scaledX / i))
        phi_X.append(row)
  elif part=='d':
    # DONE
    for scaledX in X:
        row = [1]
        for i in range(1,50):
            row.append(cos(scaledX / i))
        phi_X.append(row)

  return phi_X

  # Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,y):
    w_star = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return w_star

_, ax = plt.subplots(2,2, figsize = (16,10))

for i, part in enumerate(['a', 'b', 'c' ,'d']):
  # Plotting the original data
  
  ax[i//2, i%2].scatter(year_train, temp_train, label = "Original Data")
  ax[i//2, i%2].set_xlabel("Year")
  ax[i//2, i%2].set_ylabel("Temperature")
  ax[i//2, i%2].set_title(f"OLS Basis Regression; Temperature on Years ({part})")
  ax[i//2, i%2].invert_xaxis()

  xs = np.linspace(year_train.min(), year_train.max(), 1000)

  # DONE: plot your functions for the specified xs
  weights = find_weights(np.matrix(make_basis(year_train, part)), np.matrix(temp_train).T)
  y_pred = np.matmul(weights.T, np.matrix(make_basis(xs, part)).T).T
  ax[i//2, i%2].plot(xs, y_pred, color = 'orange', label = "Basis Regression")

  ax[i//2, i%2].legend()

plt.savefig("images/p3.1.png")
  

def mean_squared_error(X, y, w):
  # DONE: Given a linear regression model with parameter w, compute and return the
  # mean squared error.
    X, y, w = map(np.matrix, (X, y, w))
    y_pred = np.matmul(w.T, X.T).T
    y_pred, y = map(np.array, (y_pred, y))

    mse = 0
    for pred, val in zip(y_pred, y[0]):
       mse += (val - pred[0]) ** 2
    
    return mse / len(y_pred)


from math import sqrt

def negative_log_likelihood(X,y,w, sigma):
  # DONE: Given a probabilistic linear regression model y = w^T x + e, where
  # e is N(0, sigma), return the negative log likelihood of the data X,y.
  
  N = float(np.array(X).shape[0])
  mse = mean_squared_error(X, y, w)

  likelihood = N * (np.log(np.sqrt(2 * np.pi) * sigma) + mse / (2 * sigma ** 2))

  return likelihood


for part in ['a', 'b', 'c', 'd']:
  # DONE: compute the MSE and Likelihood and print the results
  xs_train_basis = make_basis(year_train, part)
  weights_star = find_weights(np.matrix(xs_train_basis), np.matrix(temp_train).T)
  y_train_pred = np.matmul(weights_star.T, np.matrix(xs_train_basis).T).T

  xs_test_basis = make_basis(year_test, part)
  y_test_pred = np.matmul(weights_star.T, np.matrix(xs_test_basis).T).T

  sigma_mle = sqrt(mean_squared_error(make_basis(year_train, part), temp_train, weights_star))
  print(f"Sigma is {sigma_mle}\n")

  train_mse = mean_squared_error(xs_train_basis,temp_train,weights_star)
  test_mse = mean_squared_error(xs_test_basis,temp_test,weights_star)
  
  print(f"\nPart ({part});\n\n Train MSE: {train_mse}; Test MSE: {test_mse}\n")
  
  # DONE: compute the likelihood.
  train_log_nll = negative_log_likelihood(xs_train_basis,temp_train,weights_star,sigma_mle)
  test_log_nll = negative_log_likelihood(xs_test_basis,temp_test,weights_star,sigma_mle)
  print(f" Train Negative Log-Likelihood: {train_log_nll:.3f}; Test Negative Log-Likelihood: {test_log_nll:.3f}")