import numpy as np

def matrix_image(A):
    # convert to row echelon form
    # find pivot columns
    # return pivot columns (from original matrix)
    A = A.astype(float)
    rows, columns = A.shape
    row_echelon_matrix = A.copy()
    pivot_column = []
    pivot_row = 0 # pivot row

    for col in range(columns):
        if pivot_row >= rows: break

        # finding pivot  
        pivot = np.argmax(np.abs(row_echelon_matrix[pivot_row:, col])) + pivot_row # look below and including current pivot row
        if np.isclose(row_echelon_matrix[pivot, col], 0):
            continue # no pivot exists: move to next col 

        # swap rows to bring pivot into position
        row_echelon_matrix[[pivot_row, pivot]] = row_echelon_matrix[[pivot, pivot_row]]
        # make pivot val 1
        row_echelon_matrix[pivot_row] /= row_echelon_matrix[pivot_row, col]
        # elimnate entries below pivot (ri -> ri - (ri,c) * r_r)

        for i in range(pivot_row + 1, rows):
            row_echelon_matrix[i] -= row_echelon_matrix[i, col] * row_echelon_matrix[pivot_row]

        pivot_column.append(col)
        pivot_row += 1

    return A[:, pivot_column]


if __name__ == "__main__":

    matrix = np.array([ [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ])

    print(matrix_image(matrix))

