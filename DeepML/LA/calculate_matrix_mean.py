# https://www.deep-ml.com/problems/4

"""
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode.
The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

Example:
Input:
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
Output:
    [4.0, 5.0, 6.0]

Reasoning:
    Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].

"""


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    if mode == "row":
        for r in matrix:
            means.append(sum(r) / len(r))
    elif mode == "column":
        num_columns = len(matrix[0])
        for j in range(num_columns):
            column = [matrix[i][j] for i in range(len(matrix))]
            means.append(sum(column) / len(matrix))  # divide by row
    else:
        return []
    return means


# zip(*matrix) effectively transposes the matrix — turning columns into rows — so you can loop through columns just like rows.
# The * operator unpacks matrix so that each row is passed as an individual argument to zip. This means:


def calc(matrix, mode):
    if mode == "row":
        return [sum(r) / len(r) for r in matrix]
    elif mode == "column":
        return [sum(c) / len(c) for c in zip(*matrix)]
    else:
        return []


def main():
    print(calculate_matrix_mean(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode="row"))
    print(calc(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode="row"))


main()
