import numpy as np


class TestVoidCreaction:
    def test_incompatible_void_to_object(self):
        s1 = np.zeros(1, dtype="i,i")[0]
        s2 = np.zeros(1, dtype="i,i,i")[0]
        assert np.array([s1, s2]).dtype == np.dtype("O")

    def test_passed_in_void(self):
        # TODO: Find out why this does not fail :)
        s1 = np.zeros(1, dtype="i,i")[0]
        s2 = np.zeros(1, dtype="i,i,i")[0]
        assert np.array([s1], dtype="V").dtype == np.dtype("V8")
        size = np.dtype("O").itemsize
        out_dtype = np.dtype("V{}".format(size))
        assert np.array([s1, s2], dtype="V").dtype == out_dtype
        assert np.array([s2, s2], dtype="V").dtype == np.dtype("V12")
