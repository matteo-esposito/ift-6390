import numpy as np

# %%
"""
# IFT6390 - Lab Midterm Instructions

You have exactly one hour to take this exam.

You need to submit your solutions as a `.py` file named `solution.py` to the Gradescope entry `Lab Midterm`.

Each function or method you are asked to implement is worth 1 point. 
20% of the marks are given to visible autograder test cases for which the input and desired output is visible, 
and the remaining 80% of the points are given to autograder test cases for which you would not see the input and output.
"""


# %%
"""
# Python basics
"""


# %%
"""
## 1 - Favorite string

Alice has a favorite string, `fav_string`. For her birthday, Alice’s friends decide to give her strings as birthday gifts. 
Alice will like a string if `fav_string` is a substring of it. 
A substring is a string formed by keeping contiguous caracters of the original string and removing the remaining characters around it (could remove zero characters).

Given a list of strings, how many of them will Alice like?

---

Example:

`fav_string='cod', input_list=['coding', 'crocodilian', 'doc', 'recodification']`

`Output= 3`
"""


# %%
def count_fav_strings(fav_string, input_list):
    """
    :fav_string: string
    :input_list: list of strings
    :return: int
    """
    pass


# %%
"""
## 2 - Strange list

$n$ numbers are written on the board. Alice and Bob decide to play a game to create a list from the numbers.
First, Alice selects the largest number on the board and places it in the first element of the list. Bob then picks the smallest number and sets it as the second element of the list. Now Alice picks the largest among the remaining numbers on the board and places it in the third element of the list. The game continues until there is no number remaining on the board.
Your function should take the list of unorganized numbers and return Alice and Bob's new list.

---

Example:

`input_list=[2, 5, 2, 7, 1, 6, 4]`
 
`output_list=[7, 1, 6, 2, 5, 2, 4]`
"""


# %%
def strange_list(input_list):
    """
    :input_list: list[float]
    :return: list[float]
    """    
    pass


# %%
"""
## 3 - Robust mean
Given a list of numbers $(a_1, ...., a_n)$, return the average of all elements except for the largest element and the smallest element. If there are multiple elements that have the maximal or minimal value, you should remove from the list exactly one maximal and exactly one minimal element before taking the average. For lists with two elements you should return the normal average, and for lists with less than two elements you should return None.

---

Example:

`input_list=[1,2,3,4,10]`

`Output= 3`
"""


# %%
def robust_mean(input_list):
    """
    :input_list: list[float]
    :return: float
    """     
    pass


# %%
"""
## 4 - Steps to equal

Alice and Bob have three bowls which contain a,b,c liters of water each. They need to equalize the amount of water in all three bowls. To make this tedious chore appealing, they decide to play a game.
In each step of the game, they pick two bowls and pour some amount of water from one bowl into the other. What is the minimum number of steps needed to equalize the amount of water in all three bowls?

You can consider that the bowls can contain any quantity of water.

---

Example 1:

`a=10, b=10, c=10`

`Output=0`
 
Example 2:

`a=1, b=1000, c=20`

`Output=2`
"""


# %%
def steps_to_equal(a, b, c):
    """
    :a: float
    :b: float
    :c: float
    :return: int
    """ 
    pass


# %%
"""
# Numpy
"""


# %%
"""
## 5 - Missing values

Sometimes real datasets are missing some of their values, because some features for some of the examples have not been collected/measured.
When dealing with missing values we sometimes represent them as zeros in a dataset, as long as zero is not a meaningful value in the dataset (e.g., peoples weights and heights, house prices, etc.). 

In this question, zeros in a dataset represent missing values.
 
a. Given a dataset X (`np.array` of size N x d), with missing values represented as zeros, remove the samples (rows) with at least one missing value.

Example:

`X=np.array([[1,2,0], [4,5,6], [7,8,9]])`

`Output= np.array([[4,5,6], [7,8,9]])`

---

b. Given a dataset X (`np.array` of size N x d), with missing values represented as zeros, remove the features (columns) with at least one missing value.

Example:

`X=np.array([[1,2,0], [4,5,6], [7,8,9]])`

`Output= np.array([[1,2], [4,5], [7,8]])`

---

c. Given a dataset X (`np.array` of size N x d), with missing values represented as zeros, replace the missing values with the empirical average value of the particular feature in the training set. To compute that empirical average, you should consider only the examples for which the particular feature is not missing.

Example:

`X=np.array([[1,2,0], [4,5,6], [7,8,9]])`

`Output= np.array([[1,2,7.5], [4,5,6], [7,8,9]])`
"""


# %%
def remove_missing_samples(X):
    """
    :param X: float np.array of size N x d (each row is a data point)
    :return: float np.array
    """
    pass


# %%
def remove_missing_features(X):
    """
    :param X: float np.array of size N x d (each row is a data point)
    :return: float np.array
    """
    pass


# %%
def interpolate_missing_values(X):
    """
    :param X: float np.array of size N x d (each row is a data point)
    :return: float np.array
    """
    pass


# %%
"""
## 6 - Moving average

A common technique used for smoothing data is the moving average. Given a vector $x \in \mathbb{R}^d$ the $k$-moving average is another $d$-dimensional vector defined as follows:
* for $i \geq k$ it is the average of the latest $k$ values (up to the $i$-th position) of the original sequence, i.e., $out_i = \frac{\sum_{j=1}^k x_{i+j-k}}{k}$. 
* for $i < k$,  to make the moving average the same size of the input ($x$), the output elements for $i < k$ will be the same as the input, i.e., $out_i=x_i$.


Implement the moving average of a 1-D `np.array`.

---

Example:

`x=np.array([1,2,3,4])`, `k=2`

`Output= np.array([1,1.5,2.5,3.5])`
"""


# %%
def moving_average(x, k):
    """
    :param x: float np.array of size d
    :param k: int
    :return: float np.array of size d
    """    
    pass


# %%
"""
# Machine Learning
"""


# %%
"""
## 7 - Perceptron algorithm

The perceptron algorithm is a simple algorithm for learning a linear classifier, i.e., the prediction rule for a data point $x$ is given by:
$$
y = 
\begin{cases} 
    1  & w^T x > 0 \\
    -1 & \text{otherwise}
\end{cases}
$$
where $w$ is the weight vector.

Starting with a weight vector $(w_0)$ initialized with zeros, at each stage, the algorithm goes over the data points and updates the weights for the first example it mislabeled. 
The algorithm terminates when all the training examples are labeled correctly.   

The weight update rules are the followings:

Mistake on a positive example ($x$): $w_{t+1} = w_t + \eta x$

Mistake on a negative example ($x$): $w_{t+1} = w_t - \eta x$

where $\eta$ is the learning rate.

Complete the methods for the `Perceptron` class provided below.

Note: we assume the class labels are -1 or +1.
"""


# %%
class Perceptron():
    def __init__(self, d, learning_rate):
        """Initialize a d dimensional Perceptron classifier

        :param d: int
        :param lr: float
        """
        self.d = d
        self.learning_rate = learning_rate
        self.w = np.zeros(d)

    def predict(self, X):
        """For each data point in the input X, return the predicted class

        :param X: float np.array of size N x d (each row is a data point)
        :return: int np.array of size N (predicted class of each data point -1 or +1)
        """
        pass

    def fit(self, X, y):
        """Train the weights of the classifier

        :param X: float np.array of size N x d (each row is a data point)
        :param y: int np.array of size N (class of each data point -1 or +1)
        """
        pass