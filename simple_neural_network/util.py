# simple_neural_network.py

""" Follows Chapter 7 of Classic Computer Science Problems in Python by David Kopec
"""

from typing import List
from math import exp

# compute the dot product of two functions
def dot_product(xs: List[float], ys: List[float]) -> float:
    return sum(x * y for x, y in zip(xs,ys))

# the classic sigmoid activation function
def sigmoid(x: float) -> float:
	return 1.0 / (1.0 + exp(-x))

def derivative_sigmoid(x: float) -> float:
	sig: float = sigmoid(x)
	return sig * ( 1 - sig)

"""
x = [1, 2, 3]


print(f"{dot_product(x,x)=}")

for i in range(10):
	print(f"sigmoid(10 ** {i}) = {sigmoid(10 ** i)})")

print(f"{derivative_sigmoid(2)=}")
"""

# assume all rows are of equal length
# and feature scale each column to be in the range 0 - 1
def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
    for col_num in range(len(dataset[0])):
        column: List[float] = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        for row_num in range(len(dataset)):
            dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)


my_dataset = [[1.0,10,3,4,5], [2.0,30.0,4,5, 10],[1,3,5,6,20], [3,7,10,6,20]]
normalize_by_feature_scaling(my_dataset)
print(my_dataset)
