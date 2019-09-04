#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "abstractdtype.h"


NPY_NO_EXPORT int
init_pyvalue_abstractdtypes()
{
    PyType_Ready((PyTypeObject *)&PyArray_PyIntAbstractDType);
    PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType);
}


// TODO: There are two possible designs here (or both), the one below is to
//       to make the DType class "instances" subclasses of a given class which
//       is exposed and allows to attach the information of creating new such
//       subclasses to it. The other thing is that we may want a cache of
//       these classes, since they are only used very locally (hidden) and
//       we should never really use much more than 2, or at least a few,
//       at a time.
//       The last means that it is necessary to create a new MetaClass to
//       override the dealloc. From the python side some of the information
//       could be exposed as classmethods in either case. From the C-side
//       we may provide functions to deal with these.
//       (say an Int24 wants to promote correctly with it).

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


