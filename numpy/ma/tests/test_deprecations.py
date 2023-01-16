"""Test deprecation and future warnings.

"""
import pytest
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
import io
import textwrap


class TestFromtextfile:
    def test_fromtextfile_delimitor(self):
        # NumPy 1.22.0, 2021-09-23

        textfile = io.StringIO(textwrap.dedent(
            """
            A,B,C,D
            'string 1';1;1.0;'mixed column'
            'string 2';2;2.0;
            'string 3';3;3.0;123
            'string 4';4;4.0;3.14
            """
        ))

        with pytest.warns(DeprecationWarning):
            result = np.ma.mrecords.fromtextfile(textfile, delimitor=';')
