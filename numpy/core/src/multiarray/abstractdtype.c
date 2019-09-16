#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "abstractdtype.h"


static PyArray_DTypeMeta *
common_dtype_int(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    PyObject *maximum = ((PyArray_PyValueAbstractDType *)cls)->maximum;
    PyObject *minimum = ((PyArray_PyValueAbstractDType *)cls)->minimum;

    PyObject *max_other = NULL;
    PyObject *min_other = NULL;

    int res_max, res_min;

    /* Lets just use direct slot lookup for now: */
    // TODO: The second cast should be unnecessary, since we need to expose it :(?
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyIntAbstractDType) {
        /* Modify ourself in-place and return */
        max_other = ((PyArray_PyValueAbstractDType *)other)->maximum;
        min_other = ((PyArray_PyValueAbstractDType *)other)->minimum;

        Py_INCREF(max_other);
        Py_INCREF(min_other);

        goto return_other_or_updated_minmax;
    }
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyFloatAbstractDType) {
        Py_INCREF(other);
        return other;
    }
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyComplexAbstractDType) {
        Py_INCREF(other);
        return other;
    }

    /* Quickly return not implemented, if we cannot handle things */
    if (other->type_num < 0 || other->type_num >= NPY_USERDEF) {
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }

    /* Handle integers, a bit non-elegant, but user types are rejected above */
    if ((other->kind == 'i') || (other->kind == 'u')) {
        npy_ulonglong maximum = 0;
        npy_longlong minimum = 0;
        switch (other->type_num) {

        case NPY_BYTE:
            maximum = NPY_MAX_BYTE;
            minimum = NPY_MIN_BYTE;
            break;
        case NPY_SHORT:
            maximum = NPY_MAX_SHORT;
            minimum = NPY_MIN_SHORT;
            break;
        case NPY_INT:
            maximum = NPY_MAX_INT;
            minimum = NPY_MIN_INT;
            break;
        case NPY_LONG:
            maximum = NPY_MAX_LONG;
            minimum = NPY_MIN_LONG;
            break;
        case NPY_LONGLONG:
            maximum = NPY_MAX_LONGLONG;
            minimum = NPY_MIN_LONGLONG;
            break;
        /* Unsigned versions: */
        case NPY_UBYTE:
            maximum = NPY_MAX_UBYTE;
            break;
        case NPY_USHORT:
            maximum = NPY_MAX_USHORT;
            break;
        case NPY_UINT:
            maximum = NPY_MAX_UINT;
            break;
        case NPY_ULONG:
            maximum = NPY_MAX_ULONG;
            break;
        case NPY_ULONGLONG:
            maximum = NPY_MAX_ULONGLONG;
            break;
        default:
            assert(0);  /* Cannot happen */
        }

        max_other = PyLong_FromUnsignedLongLong(maximum);
        if (max_other == NULL) {
            return NULL;
        }
        min_other = PyLong_FromLongLong(minimum);
        if (min_other == NULL) {
            Py_DECREF(max_other);
            return NULL;
        }

        goto return_other_or_updated_minmax;
    }

    /* NOTE: We should not normally get this far. */
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;

    // TODO: Maybe should likely make this a function rather than a goto...
return_other_or_updated_minmax:
    res_max = PyObject_RichCompareBool(maximum, max_other, Py_GT);
    if (res_max < 0) {
        goto fail;
    }
    res_min = PyObject_RichCompareBool(minimum, min_other, Py_LT);
    if (res_min == -1) {
        goto fail;
    }
    if (!res_min && !res_max) {
        /* The other type is capable of holding both min and max: */
        Py_DECREF(max_other);
        Py_DECREF(min_other);
        Py_INCREF(other);
        return other;
    }
    /* The other type is not capable of holding both, so update self/cls */
    if (res_max) {
        Py_INCREF(max_other);
        ((PyArray_PyValueAbstractDType *)cls)->maximum = max_other;
        Py_DECREF(maximum);
    }
    if (res_min) {
        Py_INCREF(min_other);
        ((PyArray_PyValueAbstractDType *)cls)->minimum = min_other;
        Py_DECREF(minimum);
    }
    Py_DECREF(max_other);
    Py_DECREF(min_other);
    Py_INCREF(cls);
    return cls;

fail:
    Py_XDECREF(max_other);
    Py_XDECREF(min_other);
    return NULL;
}

static PyArray_DTypeMeta *
default_dtype_int(PyArray_DTypeMeta *cls) {
    // TODO: May need to return an Object here in some cases?
    PyArray_DTypeMeta *dtype = (PyArray_DTypeMeta *)Py_TYPE(PyArray_DescrFromType(NPY_LONG));
    Py_INCREF(dtype);
    return dtype;
}


static PyArray_DTypeMeta *
minimal_dtype_int(PyArray_DTypeMeta *cls) {
    // TODO: This is incorrect, I need to return the actual minimal type.
    // TODO: May need to think about the need of this, it may need to be
    //       more specific for integers, at which point
    //       the whole slot may be useless...
    PyArray_DTypeMeta *dtype = (PyArray_DTypeMeta *)Py_TYPE(PyArray_DescrFromType(NPY_LONG));
    Py_INCREF(dtype);
    return dtype;
}



static PyArray_DTypeMeta*
discover_dtype_from_pyint(PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    // TODO: There probably needs to be some optimizations here...
    PyArray_DTypeMeta *dtype = PyObject_New(
            PyArray_PyValueAbstractDType, &PyArrayAbstractObjDTypeMeta_Type);
    if (dtype == NULL) {
        return NULL;
    }

    // TODO: Does this break a few fields in principle?
    /* Copy most things from the base type, since it is the same */
    memcpy((char *)dtype + sizeof(PyObject),
           (char *)(&PyArray_PyIntAbstractDType) + sizeof(PyObject),
           sizeof(PyArray_PyValueAbstractDType) -  sizeof(PyObject));
    ((PyTypeObject*)dtype)->tp_base = (PyTypeObject *)&PyArray_PyIntAbstractDType;
    ((PyTypeObject*)dtype)->tp_name = "numpy.PyIntAbstractDType";
    Py_INCREF(&PyArrayAbstractObjDTypeMeta_Type);

    if (PyType_Ready((PyTypeObject*)dtype) < 0) {
        Py_DECREF(dtype);
        return NULL;
    }

    PyType_Slot slots[] = {
        {NPY_dt_common_dtype, common_dtype_int},
        {NPY_dt_default_dtype, default_dtype_int},
        {NPY_dt_minimal_dtype, minimal_dtype_int},
        {0, NULL},
    };

    PyArrayDTypeMeta_Spec spec = {
        .flexible = 0,
        .abstract = 1,
        .itemsize = -1,
        .flags = 0,
        .typeobj = NULL,
        .slots = slots,
    };

    if (PyArray_InitDTypeMetaFromSpec(dtype, &spec) < 0) {
        Py_DECREF(dtype);
        return NULL;
    }

    Py_INCREF(obj);
    ((PyArray_PyValueAbstractDType *)dtype)->minimum = obj;
    Py_INCREF(obj);
    ((PyArray_PyValueAbstractDType *)dtype)->maximum = obj;

    return dtype;
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
    Py_INCREF(&PyArray_PyIntAbstractDType);
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
        {NPY_dt_discover_dtype_from_pytype, discover_dtype_from_pyint},
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
    PyArray_PyIntAbstractDType.super.dt_slots->requires_pyobject_for_discovery = 1;

    /* Prepare the abstract dtype used for float (value based) promotion */
    PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType);
    Py_INCREF(&PyArray_PyFloatAbstractDType);
    pytypes_float = PyTuple_Pack(5,
        &PyFloat_Type, &PyHalfArrType_Type, &PyFloatArrType_Type,
        &PyDoubleArrType_Type, &PyLongDoubleArrType_Type);
    if (pytypes_float == NULL) {
        goto fail;
    }
    PyType_Slot float_slots[] = {
        //{NPY_dt_associated_python_types, pytypes_float},
        //{NPY_dt_discover_dtype_from_pytype, discover_float},
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
    PyArray_PyFloatAbstractDType.super.dt_slots->requires_pyobject_for_discovery = 1;

    /* Prepare the abstract dtype used for float (value based) promotion */
    PyType_Ready((PyTypeObject *)&PyArray_PyComplexAbstractDType);
    Py_INCREF(&PyArray_PyComplexAbstractDType);
    pytypes_complex = PyTuple_Pack(4,
        &PyComplex_Type, &PyCFloatArrType_Type,
        &PyCDoubleArrType_Type, &PyCLongDoubleArrType_Type);
    if (pytypes_complex == NULL) {
        goto fail;
    }
    PyType_Slot complex_slots[] = {
        //{NPY_dt_associated_python_types, pytypes_complex},
        //{NPY_dt_discover_dtype_from_pytype, discover_complex},
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
    PyArray_PyComplexAbstractDType.super.dt_slots->requires_pyobject_for_discovery = 1;

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
    PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyIntBaseAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyFloatAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyFloatBaseAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
};

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyComplexAbstractDType = {{{{
    PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    .tp_name = "numpy.PyComplexBaseAbstractDType",
    .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,  /* Real or complex part minimum/maximum... */
    .maximum = NULL,
};

