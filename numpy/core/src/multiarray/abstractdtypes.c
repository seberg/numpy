#include <numpy/ndarraytypes.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"

#include "abstractdtypes.h"
#include "array_coercion.h"
#include "common.h"


/*
 * Value Based Promotion Notes
 * ---------------------------
 *
 * One thing we have to solve here is "value based promotion" which is not
 * a walk in the park...
 *
 * Value based promotion in NumPy is limited to:
 *
 *   * A _mix_ of scalars and arrays.
 *   * Should only be used for Python scalars (As of NumPy 1.20 not the case).
 *
 * The first point is important, because it means that when we have only
 * scalars with the respective abstract DTypes (defined here largely), we
 * do not need to promote them at all.  We can instead convert them to their
 * "default" DType and promote those instead.
 *
 * The second point is mainly important because it narrows down when value
 * based promotion is used. For example `np.add(1, [2, 3])` will never consider
 * the list `[2, 3]` as a "value" and can convert it using `np.asarray()`
 * without checking for value based promotion.
 *
 * This means we can implement the following scheme:
 *   1. We first promote all concrete DTypes, those which do not come with a
 *      scalar value. (Users will have to be careful to do this also.)
 *   2. If there is no concrete DType, we convert all of the abstract ones
 *      to their respective concrete version (using the provided value)
 *      and perform 1. on those.
 *   3. If we have a mix, there will be a single concrete DType left after
 *      step 1.  We promote this _with_ value with the first abstract DType.
 *      The result must be concrete again, and the process is continued until
 *      we have the final, concrete, result.
 *
 * To implement this scheme, we have to pass around scalar values for abstract
 * DTypes (note that if an abstract DType does not wish to take part in value
 * based promotion, it will always be converted to the concrete version right
 * away, `np.result_type`, etc. will reject this case).
 * This means we must pass scalars into the C-equivalent of `np.result_types`
 * as well as the universal function promoters.
 *
 * In NumPy 1.20, only the abstract DType will have a slot to actually perform
 * value based promotion. But to allow value based promotion for user DTypes
 * this slot will need to be public. In that case the slot of the concrete
 * DType will be guaranteed to be called first (if it is defined).
 *
 * In NumPy 1.20 we reuse `discover_descriptor_from_pyobject()` even when
 * we may only require the type of that descriptor. A dedicated function
 * may be preferable if we allow users to create such "value sensitive"
 * abstract DTypes. (Investigate how and where would use it.)
 */


NPY_NO_EXPORT PyArray_DTypeMeta *  // TODO: oops, static!
pyint_common_dtype_with_value(PyArray_DTypeMeta *self,
        PyArray_DTypeMeta *other, PyObject *value)
{
    if (other->type_num > NPY_NTYPES ) {
        /*
         * Return NotImplemented, indicating that the promotion machinery
         * should make one last attempt by discovering the correct concrete
         * DType from the value.
         */
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta *)Py_NotImplemented;
    }
    return NULL;
}


static PyArray_Descr *
discover_descriptor_from_pyint(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyLong_Check(obj));
    /*
     * We check whether long is good enough. If not, check longlong and
     * unsigned long before falling back to `object`.
     */
    long long value = PyLong_AsLongLong(obj);
    if (error_converting(value)) {
        PyErr_Clear();
    }
    else {
        if (NPY_MIN_LONG <= value && value <= NPY_MAX_LONG) {
            return PyArray_DescrFromType(NPY_LONG);
        }
        return PyArray_DescrFromType(NPY_LONGLONG);
    }

    unsigned long long uvalue = PyLong_AsUnsignedLongLong(obj);
    if (uvalue == (unsigned long long)-1 && PyErr_Occurred()){
        PyErr_Clear();
    }
    else {
        return PyArray_DescrFromType(NPY_ULONGLONG);
    }

    return PyArray_DescrFromType(NPY_OBJECT);
}


static PyArray_Descr*
discover_descriptor_from_pyfloat(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyFloat_CheckExact(obj));
    return PyArray_DescrFromType(NPY_DOUBLE);
}


static PyArray_Descr*
discover_descriptor_from_pycomplex(
        PyArray_DTypeMeta* NPY_UNUSED(cls), PyObject *obj)
{
    assert(PyComplex_CheckExact(obj));
    return PyArray_DescrFromType(NPY_COMPLEX128);
}


NPY_NO_EXPORT int
initialize_and_map_pytypes_to_dtypes()
{
    ((PyTypeObject *)&PyArray_PyIntAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
    PyArray_PyIntAbstractDType.scalar_type = &PyLong_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyIntAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyFloatAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
    PyArray_PyFloatAbstractDType.scalar_type = &PyFloat_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyFloatAbstractDType) < 0) {
        return -1;
    }
    ((PyTypeObject *)&PyArray_PyComplexAbstractDType)->tp_base = &PyArrayDTypeMeta_Type;
    PyArray_PyComplexAbstractDType.scalar_type = &PyComplex_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_PyComplexAbstractDType) < 0) {
        return -1;
    }

    /* Register the new DTypes for discovery */
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyIntAbstractDType, &PyLong_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyFloatAbstractDType, &PyFloat_Type, NPY_FALSE) < 0) {
        return -1;
    }
    if (_PyArray_MapPyTypeToDType(
            &PyArray_PyComplexAbstractDType, &PyComplex_Type, NPY_FALSE) < 0) {
        return -1;
    }

    /*
     * Map str, bytes, and bool, for which we do not need abstract versions
     * to the NumPy DTypes. This is done here using the `is_known_scalar_type`
     * function.
     * TODO: The `is_known_scalar_type` function is considered preliminary,
     *       the same could be achieved e.g. with additional abstract DTypes.
     */
    PyArray_DTypeMeta *dtype;
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_UNICODE));
    if (_PyArray_MapPyTypeToDType(dtype, &PyUnicode_Type, NPY_FALSE) < 0) {
        return -1;
    }

    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_STRING));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBytes_Type, NPY_FALSE) < 0) {
        return -1;
    }
    dtype = NPY_DTYPE(PyArray_DescrFromType(NPY_BOOL));
    if (_PyArray_MapPyTypeToDType(dtype, &PyBool_Type, NPY_FALSE) < 0) {
        return -1;
    }

    return 0;
}



NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyIntAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._PyIntBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pyint,
    .kind = 'i',
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyFloatAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
       .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._PyFloatBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pyfloat,
    .kind = 'f',
};

NPY_NO_EXPORT PyArray_DTypeMeta PyArray_PyComplexAbstractDType = {{{
        PyVarObject_HEAD_INIT(&PyArrayDTypeMeta_Type, 0)
        .tp_basicsize = sizeof(PyArray_Descr),
         .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_name = "numpy._PyComplexBaseAbstractDType",
    },},
    .abstract = 1,
    .discover_descr_from_pyobject = discover_descriptor_from_pycomplex,
    .kind = 'c',
};
