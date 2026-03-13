# https://www.deep-ml.com/problems/121?returnTo=paths

def vector_sum(a: list[int | float], b: list[int | float]) -> list[int | float] | int:
    # If vectors have different lengths, return -1
    if len(a) != len(b):
        return -1

    # Element-wise sum
    return [x + y for x, y in zip(a, b)]
