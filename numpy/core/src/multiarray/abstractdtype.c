#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "abstractdtype.h"


static PyArray_DTypeMeta*
discover_dtype_from_pyint(PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    // TODO: There probably needs to be some optimizations here...


    Py_INCREF(obj);
    ->minimum = obj;
    Py_INCREF(obj);
    ->maximum = obj;
}


NPY_NO_EXPORT int
init_pyvalue_abstractdtypes()
{
    PyObject *pytypes_int = NULL;
    PyObject *pytypes_float = NULL;
    PyObject *pytypes_complex = NULL;
    // TODO: Should i make macros for this, it is almost identical code?

    /* Prepare the abstract dtype used for integer (value based) promotion */
    PyType_Ready((PyTypeObject *)&PyArray_PyIntAbstractDType);
    /* All the types associated with Integers, are python ints and our ints */
    pytypes_int = PyTuple_Pack(11,
        &PyLong_Type, &PyByteArrType_Type, &PyShortArrType_Type,
        &PyIntArrType_Type, &PyLongArrType_Type, &PyLongLongArrType_Type,
        &PyUByteArrType_Type, &PyUShortArrType_Type,
        &PyUIntArrType_Type, &PyULongArrType_Type, &PyULongLongArrType_Type);
    if (pytypes_int == NULL) {
        goto fail;
    }
    PyType_Slot int_slots[] = {
        {NPY_dt_associated_python_types, pytypes_int},
        {NPY_dt_discover_dtype_from_pytype, discover_integer},
        {0, NULL},
    };
    PyArrayDTypeMeta_Spec int_spec = {
        .flexible = 0,
        .abstract = 1,
        .itemsize = -1,
        .flags = 0,
        .typeobj = NULL,
        .slots = int_slots,
    };
    if (PyArray_InitDTypeMetaFromSpec(
                (PyArray_DTypeMeta *)&PyArray_PyIntAbstractDType,
                &int_spec) < 0) {
        goto fail;
    }
    PyArray_PyIntAbstractDType->dt_slots->requires_pyobject_for_discovery = 1;

    /* Prepare the abstract dtype used for float (value based) promotion */
    PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType);
    pytypes_float = PyTuple_Pack(5,
        &PyFloat_Type, &PyHalfArrType_Type, &PyFloatArrType_Type,
        &PyDoubleArrType_Type, &PyLongDoubleArrType_Type);
    if (pytypes_float == NULL) {
        goto fail;
    }
    PyType_Slot float_slots[] = {
        {NPY_dt_associated_python_types, pytypes_float},
        {NPY_dt_discover_dtype_from_pytype, discover_float},
        {0, NULL},
    };
    PyArrayDTypeMeta_Spec float_spec = {
        .flexible = 0,
        .abstract = 1,
        .itemsize = -1,
        .flags = 0,
        .typeobj = NULL,
        .slots = float_slots,
    };
    if (PyArray_InitDTypeMetaFromSpec(
                (PyArray_DTypeMeta *)&PyArray_PyFloatAbstractDType,
                &float_spec) < 0) {
        goto fail;
    }
    PyArray_PyFloatAbstractDType->dt_slots->requires_pyobject_for_discovery = 1;

    /* Prepare the abstract dtype used for float (value based) promotion */
    PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType);
    pytypes_complex = PyTuple_Pack(4,
        &PyComplex_Type, &PyCFloatArrType_Type,
        &PyCDoubleArrType_Type, &PyCLongDoubleArrType_Type);
    if (pytypes_complex == NULL) {
        goto fail;
    }
    PyType_Slot complex_slots[] = {
        {NPY_dt_associated_python_types, pytypes_complex},
        {NPY_dt_discover_dtype_from_pytype, discover_complex},
        {0, NULL},
    };
    PyArrayDTypeMeta_Spec complex_spec = {
        .flexible = 0,
        .abstract = 1,
        .itemsize = -1,
        .flags = 0,
        .typeobj = NULL,
        .slots = complex_slots,
    };
    if (PyArray_InitDTypeMetaFromSpec(
                (PyArray_DTypeMeta *)&PyArray_PyComplexAbstractDType,
                &complex_spec) < 0) {
        goto fail;
    }
    PyArray_PyComplexAbstractDType->dt_slots->requires_pyobject_for_discovery = 1;

    Py_XDECREF(pytypes_int);
    Py_XDECREF(pytypes_float);
    Py_XDECREF(pytypes_complex);
    return 0;

fail:
    Py_XDECREF(pytypes_int);
    Py_XDECREF(pytypes_float);
    Py_XDECREF(pytypes_complex);
    return -1;
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
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyIntAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyFloatAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyIntAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyComplexAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyComplexAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,  /* Real or complex part minimum/maximum... */
    .maximum = NULL,
};

