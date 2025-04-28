# https://www.deep-ml.com/problems/1

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# num(col) of matrix == len of vector
	if len(a) == 0 or len(a[0]) != len(b): return -1
	res = []
	for row in a:
		dot_prod = sum(row[i] * b[i] for i in range(len(b)))
		res.append(dot_prod)
	return res


