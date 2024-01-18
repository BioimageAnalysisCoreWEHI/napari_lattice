from numpy import ndarray

def test(a, b, c):
    assert isinstance(a, ndarray)
    assert isinstance(b, str)
    assert isinstance(c, int)
    return a
