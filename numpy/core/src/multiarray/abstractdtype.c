#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "abstractdtype.h"



NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyIntAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
    .tp_name = "numpy.PyIntAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyFloatAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
    .tp_name = "numpy.PyIntAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};


