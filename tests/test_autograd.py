from algorithms.deeplearning.autograd import Value


def test_add_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_mul_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0


def test_add_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_mul_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0  # dc/da = b
    assert b.grad == 2.0  # dc/db = a


def test_relu_positive():
    a = Value(5.0)
    b = a.relu()
    assert b.data == 5.0
    b.backward()
    assert a.grad == 1.0


def test_relu_negative():
    a = Value(-5.0)
    b = a.relu()
    assert b.data == 0.0
    b.backward()
    assert a.grad == 0.0


def test_chain_rule():
    # f = (a + b) * c
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    d = (a + b) * c  # d = 20
    d.backward()
    assert d.data == 20.0
    assert a.grad == 4.0  # dd/da = c = 4
    assert b.grad == 4.0  # dd/db = c = 4
    assert c.grad == 5.0  # dd/dc = (a+b) = 5


def test_gradient_accumulation():
    # f = a * a (gradient should be 2a)
    a = Value(3.0)
    b = a + a  # b = 2a
    b.backward()
    assert a.grad == 2.0


def test_against_pytorch():
    """Verify gradients match PyTorch autograd on a small expression."""
    import torch

    # f = (a * b + c).relu()
    a_val, b_val, c_val = 2.0, -3.0, 10.0

    # Our autograd
    a = Value(a_val)
    b = Value(b_val)
    c = Value(c_val)
    d = (a * b + c).relu()
    d.backward()

    # PyTorch
    at = torch.tensor(a_val, requires_grad=True)
    bt = torch.tensor(b_val, requires_grad=True)
    ct = torch.tensor(c_val, requires_grad=True)
    dt = (at * bt + ct).relu()
    dt.backward()

    assert abs(a.grad - at.grad.item()) < 1e-6
    assert abs(b.grad - bt.grad.item()) < 1e-6
    assert abs(c.grad - ct.grad.item()) < 1e-6
