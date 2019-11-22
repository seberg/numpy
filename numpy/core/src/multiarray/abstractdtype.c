#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "abstractdtype.h"
#include "common.h"

#include "numpy/npy_math.h"
#include <numpy/npy_3kcompat.h>


/*
 * A very short cache, normally we should only have two at once really,
 * although e.g. for ufuncs more could be necessary.
 */
#define ABSTRACTDTYPE_CACHE_SIZE 4
static PyObject *pyint_abstractdtype_cache[ABSTRACTDTYPE_CACHE_SIZE] = {NULL};
static PyObject *pyfloat_abstractdtype_cache[ABSTRACTDTYPE_CACHE_SIZE] = {NULL};
static PyObject *pycfloat_abstractdtype_cache[ABSTRACTDTYPE_CACHE_SIZE] = {NULL};




/*
 * Integer Abstract DType slots:
 */


static PyArray_DTypeMeta *
common_dtype_int(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    PyObject *maximum = ((PyArray_PyValueAbstractDType *)cls)->maximum;
    PyObject *minimum = ((PyArray_PyValueAbstractDType *)cls)->minimum;

    PyObject *max_other = NULL;
    PyObject *min_other = NULL;

    int res_max, res_min;

    /* Lets just use direct slot lookup for now on the base: */
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyIntAbstractDType) {
        /* We need to find the combined minimum and maximum. */
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
        // TODO: We need to store the largest DType that
        //       was already seen (that means also when combining two of
        //       these). Because the largest non-abstract is the smallest
        //       allowed non-abstract!
        npy_ulonglong maximum_val = 0;
        npy_longlong minimum_val = 0;
        switch (other->type_num) {

        case NPY_BYTE:
            maximum_val = NPY_MAX_BYTE;
            minimum_val = NPY_MIN_BYTE;
            break;
        case NPY_SHORT:
            maximum_val = NPY_MAX_SHORT;
            minimum_val = NPY_MIN_SHORT;
            break;
        case NPY_INT:
            maximum_val = NPY_MAX_INT;
            minimum_val = NPY_MIN_INT;
            break;
        case NPY_LONG:
            maximum_val = NPY_MAX_LONG;
            minimum_val = NPY_MIN_LONG;
            break;
        case NPY_LONGLONG:
            maximum_val = NPY_MAX_LONGLONG;
            minimum_val = NPY_MIN_LONGLONG;
            break;
        /* Unsigned versions: */
        case NPY_UBYTE:
            maximum_val = NPY_MAX_UBYTE;
            break;
        case NPY_USHORT:
            maximum_val = NPY_MAX_USHORT;
            break;
        case NPY_UINT:
            maximum_val = NPY_MAX_UINT;
            break;
        case NPY_ULONG:
            maximum_val = NPY_MAX_ULONG;
            break;
        case NPY_ULONGLONG:
            maximum_val = NPY_MAX_ULONGLONG;
            break;
        default:
            assert(0);  /* Cannot happen */
        }

        max_other = PyLong_FromUnsignedLongLong(maximum_val);
        if (max_other == NULL) {
            return NULL;
        }
        min_other = PyLong_FromLongLong(minimum_val);
        if (min_other == NULL) {
            Py_DECREF(max_other);
            return NULL;
        }

        // TODO: This modifies cls/self, even if we return other, that should
        //       be OK, but just to note.
        ((PyArray_PyValueAbstractDType *)cls)->promoted = 1;
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


static int test_long(PyObject *obj)
{
    long res;
    int overflow;
    res = PyLong_AsLongAndOverflow(obj, &overflow);
    if (error_converting(res)) {
        /* Should be impossible, so fall through and trust it happens again. */
        PyErr_Clear();
        return -1;
    }
    return overflow;
}


static PyArray_DTypeMeta *
get_default_int_dtype(PyObject *minimum, PyObject *maximum)
{
    PyArray_DTypeMeta *dtype;
    PyArray_Descr *descriptor;

    if ((test_long(maximum) == 0) &&
            ((minimum == maximum) || (test_long(minimum) == 0))) {
        /* A long is sufficient (if long is long long we do extra work below) */
        descriptor = PyArray_DescrFromType(NPY_LONG);
        goto finish;
    }

    // TODO: Quite frankly, this path is bad enough (although the intp
    //       path should likely be the only one, meaning whatever is int64.)
    int overflow, unsigned_long = 0;
    long long res;
    res = PyLong_AsLongLongAndOverflow(maximum, &overflow);
    if ((overflow == 0) && error_converting(res)) {
        return NULL;
    }
    if ((overflow == 0) && (minimum != maximum)) {
        res = PyLong_AsLongAndOverflow(minimum, &overflow);
        if (error_converting(res)) {
            return NULL;
        }
    }
    if (overflow == 1) {
        /* The result might still fit into an unsigned type */
        // TODO: This is as horrible as the OBJECT fallback!
        /* AsLongAndOverflow is not available, so... */
        unsigned long long res_u = PyLong_AsUnsignedLongLong(maximum);
        if ((res_u == (unsigned long long) - 1) && PyErr_Occurred()) {
            /* Assume it is not a bad error, could check OverflowError... */
            PyErr_Clear();
            unsigned_long = 0;
        }
        else {
            unsigned_long = 1;
        }
    }

    if (unsigned_long) {
        descriptor = PyArray_DescrFromType(NPY_ULONGLONG);
    }
    else if (overflow == 0) {
        /* No overflow occurred, we use the long type */
        descriptor = PyArray_DescrFromType(NPY_LONGLONG);
    }
    else {
        /* Overflow occured, so we need to use Object */
        // TODO: This is a fallback that should be deprecated!
        descriptor = PyArray_DescrFromType(NPY_OBJECT);
    }

finish:
    dtype = NPY_DTMeta((PyObject *)descriptor);
    Py_INCREF(dtype);
    Py_DECREF(descriptor);
    return dtype;
}


static PyArray_DTypeMeta *
default_dtype_int(PyArray_DTypeMeta *cls) {
    /*
     * Use the same code as in the default version directly, checking the
     * size...
     */
    PyObject *maximum = ((PyArray_PyValueAbstractDType *)cls)->maximum;
    PyObject *minimum = ((PyArray_PyValueAbstractDType *)cls)->minimum;
    return get_default_int_dtype(minimum, maximum);
}


static PyArray_DTypeMeta *
minimal_dtype_int(PyArray_DTypeMeta *cls) {
    if (((PyArray_PyValueAbstractDType *)cls)->promoted) {
        // TODO: In this case I should return the actual minimal type!
        //       and not fall through to the default dtype.
        // TODO: This slot should be removed I think, which means the
        //       promoted logic can also be deleted.
    }
    return default_dtype_int(cls);
}


/*
 * Floating point Abstract DType slots:
 */


static PyArray_DTypeMeta *
common_dtype_float(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    PyObject *maximum = ((PyArray_PyValueAbstractDType *)cls)->maximum;
    PyObject *minimum = ((PyArray_PyValueAbstractDType *)cls)->minimum;

    PyObject *max_other = NULL;
    PyObject *min_other = NULL;

    int res_max, res_min;

    /* Lets just use direct slot lookup for now on the base: */
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyFloatAbstractDType) {
        /* We need to find the combined minimum and maximum. */
        max_other = ((PyArray_PyValueAbstractDType *)other)->maximum;
        min_other = ((PyArray_PyValueAbstractDType *)other)->minimum;

        Py_INCREF(max_other);
        Py_INCREF(min_other);

        goto return_other_or_updated_minmax;
    }
    if (((PyTypeObject *)other)->tp_base == (PyTypeObject *)&PyArray_PyIntAbstractDType) {
        Py_INCREF(cls);
        return cls;
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

    if (PyTypeNum_ISINTEGER(other->type_num)) {
        Py_INCREF(cls);
        return cls;
    }

    /* Handle floats, a bit non-elegant, but user types are rejected above */
    if (other->kind == 'f') {
        // TODO: We need to store the largest DType that
        //       was already seen (that means also when combining two of
        //       these). Because the largest non-abstract is the smallest
        //       allowed non-abstract!
        double maximum_val = 0;
        double minimum_val = 0;
        switch (other->type_num) {
            /* Note these values are legacy values and not necessarily exact */
            case NPY_HALF:
                maximum_val = -65000;
                minimum_val = 65000;
                break;
            case NPY_FLOAT:
                maximum_val = 3.4e38;
                minimum_val = -3.4e38;
                break;
            case NPY_DOUBLE:
                maximum_val = -1.7e308;
                minimum_val = 1.7e308;
                break;
            case NPY_LONGDOUBLE:
                maximum_val = NPY_INFINITY;
                minimum_val = -NPY_INFINITY;
                break;
            default:
                assert(0);  /* Cannot happen */
        }

        max_other = PyFloat_FromDouble(maximum_val);
        if (max_other == NULL) {
            return NULL;
        }
        min_other = PyFloat_FromDouble(minimum_val);
        if (min_other == NULL) {
            Py_DECREF(max_other);
            return NULL;
        }

        // TODO: This modifies cls/self, even if we return other, that should
        //       be OK, but just to note.
        ((PyArray_PyValueAbstractDType *)cls)->promoted = 1;
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
default_dtype_float(PyArray_DTypeMeta *NPY_UNUSED(cls)) {
    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    PyArray_DTypeMeta *dtype = NPY_DTMeta(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}


static PyArray_DTypeMeta *
minimal_dtype_float(PyArray_DTypeMeta *cls) {
    // TODO: This slot should be removed I think, which means the, it would
    //       only make sense as a helper for downstream functions.
    return default_dtype_float(cls);
}



/*
 * DType discovery for all Abstract DTypes.
 */

static PyArray_DTypeMeta *
shared_discover_dtype_from_pynumber(PyArray_DTypeMeta *cls,
        PyObject *obj, PyType_Slot *slots, PyObject *cache[ABSTRACTDTYPE_CACHE_SIZE])
{
    PyArray_DTypeMeta *dtype;

    /* Use a cached instance if possible (should be hit practically always) */
    for (int i = 0; i < ABSTRACTDTYPE_CACHE_SIZE; i++) {
        if (cache[i] != NULL) {
            dtype = (PyArray_DTypeMeta *)PyObject_Init(
                    cache[i],
                    &PyArrayAbstractObjDTypeMeta_Type);
            cache[i] = NULL;
            goto finish;
        }
    }

    dtype = (PyArray_DTypeMeta *)PyObject_New(
            PyArray_PyValueAbstractDType, &PyArrayAbstractObjDTypeMeta_Type);
    if (dtype == NULL) {
        return NULL;
    }

    // TODO: Does this break a few fields in principle?
    /* Copy most things from the base type, since it is the same */
    memcpy((char *)dtype + sizeof(PyObject),
           (char *)(&PyArray_PyIntAbstractDType) + sizeof(PyObject),
           sizeof(PyArray_PyValueAbstractDType) -  sizeof(PyObject));
    ((PyTypeObject*)dtype)->tp_base = (PyTypeObject *)cls;
    ((PyTypeObject*)dtype)->tp_name = cls->super.ht_type.tp_name;
    Py_INCREF(&PyArrayAbstractObjDTypeMeta_Type);

    if (PyType_Ready((PyTypeObject*)dtype) < 0) {
        Py_DECREF(dtype);
        return NULL;
    }

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

finish:
    ((PyArray_PyValueAbstractDType *)dtype)->promoted = 0;
            Py_INCREF(obj);
    ((PyArray_PyValueAbstractDType *)dtype)->minimum = obj;
    Py_INCREF(obj);
    ((PyArray_PyValueAbstractDType *)dtype)->maximum = obj;

    return dtype;
}


static PyArray_DTypeMeta *
discover_dtype_from_pyint(PyArray_DTypeMeta *cls,
        PyObject *obj, npy_bool use_minimal)
{
    if (!use_minimal) {
        /*
         * Use the same code as in the default version directly, checking the
         * size...
         */
        return get_default_int_dtype(obj, obj);
    }

    PyType_Slot slots[] = {
            {NPY_dt_common_dtype, common_dtype_int},
            {NPY_dt_default_dtype, default_dtype_int},
            {NPY_dt_minimal_dtype, minimal_dtype_int},
            {0, NULL},
    };

    return shared_discover_dtype_from_pynumber(cls, obj, slots, pyint_abstractdtype_cache);
}


static PyArray_DTypeMeta *
discover_dtype_from_pyfloat(PyArray_DTypeMeta *cls,
        PyObject *obj, npy_bool use_minimal)
{
    if (!use_minimal) {
        /* The default is to always use double precision */
        PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
        PyArray_DTypeMeta *dtype = NPY_DTMeta(descr);
        Py_INCREF(dtype);
        Py_DECREF(descr);
        return dtype;
    }

    PyType_Slot slots[] = {
            {NPY_dt_common_dtype, common_dtype_float},
            {NPY_dt_default_dtype, default_dtype_float},
            {NPY_dt_minimal_dtype, minimal_dtype_float},
            {0, NULL},
    };

    assert(PyFloat_CheckExact(obj));
    static PyObject *float_zero = NULL;
    /*
     * Note that `obj` should be 0 if the input is not finite, since
     * NaN, Inf, and -Inf are representable by any floating point type.
     */
    if (!npy_isfinite(PyFloat_AsDouble(obj))) {
        if (float_zero == NULL) {
            float_zero = PyFloat_FromDouble(0);
            if (float_zero == NULL) {
                return NULL;
            }
        }
        obj = float_zero;
    }
    return shared_discover_dtype_from_pynumber(cls,
            obj, slots, pyfloat_abstractdtype_cache);
}


static void
abstractobjdtypemeta_dealloc(PyArray_PyValueAbstractDType *self) {
    Py_XDECREF(self->minimum);
    Py_XDECREF(self->maximum);
    self->minimum = NULL;
    self->maximum = NULL;
    // TODO: Users should not be able to create these objects, how to
    //       achieve that? Public API only through functions enough?
    PyObject **cache = NULL;
    if (((PyTypeObject *)self)->tp_base ==
                (PyTypeObject *)&PyArray_PyIntAbstractDType) {
        cache = pyint_abstractdtype_cache;
    }
    else if (((PyTypeObject *)self)->tp_base ==
             (PyTypeObject *)&PyArray_PyFloatAbstractDType) {
        cache = pyfloat_abstractdtype_cache;
    }
    if (cache != NULL) {
        for (int i = 0; i < ABSTRACTDTYPE_CACHE_SIZE; i++) {
            if (cache[i] == NULL) {
                cache[i] = (PyObject *) self;
                return;
            }
        }
    }
    (&PyArrayDTypeMeta_Type)->tp_dealloc((PyObject *) self);
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
    pytypes_int = PyTuple_Pack(1, &PyLong_Type);
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
    pytypes_float = PyTuple_Pack(1, &PyFloat_Type);
    if (pytypes_float == NULL) {
        goto fail;
    }
    PyType_Slot float_slots[] = {
        {NPY_dt_associated_python_types, pytypes_float},
        {NPY_dt_discover_dtype_from_pytype, discover_dtype_from_pyfloat},
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


NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyIntAbstractDType = {
    {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy.PyIntBaseAbstractDType",
        .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
    .promoted = 0,
};


NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyFloatAbstractDType = {
    {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy.PyFloatBaseAbstractDType",
        .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,
    .maximum = NULL,
    .promoted = 0,
};


NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyComplexAbstractDType = {
    {{{
        PyVarObject_HEAD_INIT(&PyArrayAbstractObjDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_DTypeMeta),
        .tp_name = "numpy.PyComplexBaseAbstractDType",
        .tp_base = &PyArrayDescr_Type,
    },},},
    .minimum = NULL,  /* Real or complex part minimum/maximum... */
    .maximum = NULL,
    .promoted = 0,
};


NPY_NO_EXPORT PyTypeObject PyArrayAbstractObjDTypeMeta_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "numpy._AbstractObjDTypeMeta",
        .tp_basicsize = sizeof(PyArray_PyValueAbstractDType),
        /* methods */
        // TODO: Add alloc/init which always error.
        .tp_dealloc = (destructor)abstractobjdtypemeta_dealloc,
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_doc = "Helper MetaClass for value based casting AbstractDTypes.",
        .tp_base = &PyArrayDTypeMeta_Type,
        //.tp_init = (initproc)dtypemeta_init,
        //.tp_new = dtypemeta_new,
};

#undef ABSTRACTDTYPE_CACHE_SIZE