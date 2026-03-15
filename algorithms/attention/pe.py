# https://www.deep-ml.com/problems/85?from=Attention%20Is%20All%20You%20Need

import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        raise ValueError(f"position must be > 0 and d_model must be > 0, got position={position}, d_model={d_model}")
    # even indices use sine, odd use cosine
    # for posn pos and feature index i 
    # if i = even, op = sin(angle(pos,i)) else cos(angle(pos, i))
    
    pos_idx = np.arange(position, dtype=np.float32)
    dim_idx = np.arange(d_model, dtype=np.float32)
    
    pair_idx = np.floor(dim_idx/2) 
    denom = np.power(10000, (2 * pair_idx) / d_model)
    angle_rate = 1.0 / denom
    angle_rads = pos_idx[:, np.newaxis] * angle_rate[np.newaxis, :]

    pos_encoding = np.zeros_like(angle_rads)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])


    pos_encoding = pos_encoding[np.newaxis, ...]  # (1, position, d_model)
    return pos_encoding


def main():
    position = 2
    d_model = 8
    print(pos_encoding(position, d_model))

if __name__ == "__main__":
    main()

