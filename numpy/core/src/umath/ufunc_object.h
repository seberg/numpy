#ifndef _NPY_UMATH_UFUNC_OBJECT_H_
#define _NPY_UMATH_UFUNC_OBJECT_H_

#include <numpy/ufuncobject.h>

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args);

NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc);

/* strings from umathmodule.c that are interned on umath import */
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_out;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_where;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_axes;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_axis;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_keepdims;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_casting;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_order;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_dtype;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_subok;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_signature;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_sig;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_extobj;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_prepare;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_wrap;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_array_finalize;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_ufunc;
NPY_VISIBILITY_HIDDEN extern PyObject *npy_um_str_pyvals_name;


typedef struct {
        PyObject_HEAD
        /*
         * Identity for reduction, any of PyUFunc_One, PyUFunc_Zero
         * PyUFunc_MinusOne, PyUFunc_None, PyUFunc_ReorderableNone,
         * PyUFunc_IdentityValue.
         */
        int identity;

        /*
         * List of flags for each operand when ufunc is called by nditer object.
         * These flags will be used in addition to the default flags for each
         * operand set by nditer object.
         */
        npy_uint32 *op_flags;

        /*
         * List of global flags used when ufunc is called by nditer object.
         * These flags will be used in addition to the default global flags
         * set by nditer object.
         */
        npy_uint32 iter_flags;

        /* Identity for reduction, when identity == PyUFunc_IdentityValue */
        PyObject *identity_value;

} PyUFuncImplObject;

#endif
