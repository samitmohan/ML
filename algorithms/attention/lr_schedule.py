# Learning Rate Schedule from the paper
# lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
# Linearly increases LR during warmup, then decays proportional to 1/sqrt(step)

import numpy as np


def transformer_lr(step: int, d_model: int, warmup_steps: int = 4000) -> float:
    step = max(step, 1)  # avoid division by zero at step 0
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def get_lr_schedule(d_model: int, total_steps: int, warmup_steps: int = 4000) -> np.ndarray:
    return np.array([transformer_lr(s, d_model, warmup_steps) for s in range(1, total_steps + 1)])


def main():
    d_model = 512
    warmup_steps = 4000
    total_steps = 20000

    schedule = get_lr_schedule(d_model, total_steps, warmup_steps)
    peak_step = np.argmax(schedule) + 1
    print(f"d_model={d_model}, warmup={warmup_steps}")
    print(f"Peak LR: {schedule.max():.6f} at step {peak_step}")
    print(f"LR at step 1: {schedule[0]:.8f}")
    print(f"LR at step {total_steps}: {schedule[-1]:.6f}")

if __name__ == "__main__":
    main()
