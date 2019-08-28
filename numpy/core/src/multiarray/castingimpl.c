#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "castingimpl.h"



/*
 * CastingImpl type, this should be a subclass of in the future.
 */
NPY_NO_EXPORT PyTypeObject PyArrayCastingImpl_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.CastingImpl",
    .tp_basicsize = sizeof(CastingImpl),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

