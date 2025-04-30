# https://www.deep-ml.com/problems/5
'''
Write a Python function that multiplies a matrix by a scalar and returns the result.
Input:
    matrix = [[1, 2], [3, 4]], scalar = 2

Output:
    [[2, 4], [6, 8]]
'''
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # return [[x * scalar for x in row] for row in matrix]
    result = []
    for row in matrix:
        result.append([x * scalar for x in row])
    return result

def test_scalar_multiply() -> None:
    # Test case 1
    matrix = [[1, 2], [3, 4]]
    scalar = 2
    assert scalar_multiply(matrix, scalar) == [[2, 4], [6, 8]]

    # Test case 2
    matrix = [[0, -1], [1, 0]]
    scalar = -1
    assert scalar_multiply(matrix, scalar) == [[0, 1], [-1, 0]]
    

if __name__ == "__main__":
    print(test_scalar_multiply())