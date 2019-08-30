#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"


#include "castingimpl.h"
#include "dtypemeta.h"
#include "convert_datatype.h"


/*
 * Trivial dtype adjustment, simply returns new descriptors
 * NOTE: Define that out_descrs must be NULLed on error!
 */
static int adjust_two_descriptors_trivial(
        CastingImpl *self,
        PyArray_Descr *in_descrs[2],
        PyArray_Descr *out_descrs[2],
        NPY_CASTING NPY_UNUSED(casting))
{
    assert(Py_TYPE(in_descrs[0]) == (PyTypeObject *)self->from_dtype);
    
    /* Out descriptors must not hold references/be uninitialized/zeroed */
    out_descrs[0] = self->from_dtype->dt_slots->default_descr(self->from_dtype);
    if (out_descrs[0] == NULL) {
        return -1;
    }
    out_descrs[1] = self->to_dtype->dt_slots->default_descr(self->to_dtype);
    if (out_descrs[1] == NULL) {
        Py_DECREF(out_descrs[0]);
        return -1;
    }

    return 0;
}

/* Simply forwards input descriptors (any cast will be handled OK). */
static int adjust_two_descriptors_within_dtype(
        CastingImpl *self,
        PyArray_Descr *in_descrs[2],
        PyArray_Descr *out_descrs[2],
        NPY_CASTING casting)
{
    if (in_descrs[1] != NULL) {
        return adjust_two_descriptors_trivial(self, in_descrs, out_descrs, casting);
    }
    Py_INCREF(in_descrs[0]);
    out_descrs[0] = in_descrs[0];
    Py_INCREF(in_descrs[1]);
    out_descrs[1] = in_descrs[1];
    
    return 0;
}


/*
 * Full descriptor adaptation logic for flexible dtypes.
 */
static int adjust_two_descriptors_flexible(
        CastingImpl *self,
        PyArray_Descr *in_descrs[2],
        PyArray_Descr *out_descrs[2],
        NPY_CASTING casting)
{
    out_descrs[0] = NULL;
    out_descrs[1] = NULL;
    /* Do the full legacy dtype adaptation */
    if (in_descrs[1] != NULL) {
        /* Test whether casting is possible, the old style way */
        int res = PyArray_LegacyCanCastTypeTo(in_descrs[0], in_descrs[1], casting);
        if (res) {
            // TODO: For testing, could not do the byteswapping here!
            Py_INCREF(in_descrs[0]);
            out_descrs[0] = in_descrs[0];
            Py_INCREF(in_descrs[1]);
            out_descrs[1] = in_descrs[1];
            return 0;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                "invalid cast...");
            return -1;
        }
    }
    else {
        /*
         * We need to adapt the dtype, which means pomotion in old terms.
         */
        // This function wraps existing dtypes, user dtypes are never flexible
        // so this must be string, bytes, datetime, timedelta, or void.
        int from_type_num = self->from_dtype->type_num;
        int to_type_num = self->to_dtype->type_num;

        if (to_type_num == NPY_STRING || to_type_num == NPY_UNICODE ||
                PyTypeNum_ISDATETIME(to_type_num) || to_type_num == NPY_VOID) {
            PyArray_Descr *tmp = PyArray_DescrNewFromType(to_type_num);
            if (tmp == NULL) {
                PyErr_SetString(PyExc_TypeError,
                    "invalid cast...");
                return -1;
            }
            // TODO: For scalar objects we may have special logic here! ugg!
            out_descrs[1] = PyArray_AdaptFlexibleDType(NULL, in_descrs[0], tmp);
            if (out_descrs[1] == NULL) {
                PyErr_SetString(PyExc_TypeError,
                    "invalid cast...");
                return -1;
            }
            Py_INCREF(in_descrs[0]);
            out_descrs[0] = in_descrs[0];
            return 0;
        }
        // TODO: for Datetimes without a unit there may be corner cases...
        PyErr_SetString(PyExc_TypeError,
            "cannot discover correct legacy output dtype. "
            "Cast requires a specific output instance to work.");
        return -1;
    }
}


// NOTE: This API is pretty much fully internal, so that we can actually
//       change it. E.g. add a return value to out_stransfer and/or
//       change the way transferdata is handled (I would try to do that in a
//       way that allows easy wrappinig of the existing transferdata things
//       though.).
static int
get_transferfunction_legacy_fallback(
                CastingImpl *self,
                int aligned,
                npy_intp src_stride, npy_intp dst_stride,
                PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
                int move_references,
                PyArray_StridedUnaryOp **out_stransfer,
                NpyAuxData **out_transferdata,
                int *out_needs_api)
{
    /* Simply assert that we did not go wrong somewhere and fall back... */
    assert((src_dtype == NULL) ||
           (Py_TYPE(src_dtype) == (PyTypeObject *)self->from_dtype));
    assert((src_dtype == NULL) ||
           (Py_TYPE(dst_dtype) == (PyTypeObject *)self->to_dtype));
 
    int res = PyArray_LegacyGetDTypeTransferFunction(aligned,
                            src_stride, dst_stride,
                            src_dtype, dst_dtype,
                            move_references,
                            out_stransfer,
                            out_transferdata,
                            out_needs_api);

    if (res == NPY_SUCCEED) {
        return 0;
    }
    return -1;
}


/*
 * Wrap the legacy casting between the two descriptors into a CastingImpl
 * instance.
 * The second input must be identical if the casting is within the
 * same dtype (i.e. string of different lengths).
 * In that case the old style functionality will end up wrapping copyswapn
 * which should be compatible.
 */
NPY_NO_EXPORT PyObject *
castingimpl_legacynew(
        PyArray_DTypeMeta *from_dtype,
        PyArray_DTypeMeta *to_dtype)
{
    CastingImpl *casting_impl = NULL;

    casting_impl = (CastingImpl *)PyArray_malloc(sizeof(CastingImpl));
    if (casting_impl == NULL) {
        return NULL;
    }
    memset(casting_impl, 0, sizeof(CastingImpl));
    // TODO: Can Init fail? This is likely not the nicest way anyway?
    PyObject_Init((PyObject *)casting_impl, &PyArrayCastingImpl_Type);
    Py_INCREF(casting_impl);

    // TODO: Borrowed references may actually be fine here? But needs to be defined.
    casting_impl->from_dtype = from_dtype;
    casting_impl->to_dtype = to_dtype;

    if (from_dtype == to_dtype) {
        /*
         * We have to cast within a dtype, which needs to handle full alignment
         * and byte swapping.
         */
         if (from_dtype->flexible) {
            casting_impl->adjust_descriptors = adjust_two_descriptors_flexible;
         }
         else {
            casting_impl->adjust_descriptors = adjust_two_descriptors_within_dtype;
         }
    }
    else {
        /*
         * Casting is between two types, so that it is allowed to be simple,
         * i.e. the casting functions do not necessarily need byte swapping.
         */
         if (from_dtype->flexible || to_dtype->flexible) {
            casting_impl->adjust_descriptors = adjust_two_descriptors_flexible;
         }
         else {
            casting_impl->adjust_descriptors = adjust_two_descriptors_trivial;
         }
    }

    casting_impl->get_transferfunction = get_transferfunction_legacy_fallback;
    
    return (PyObject *)casting_impl;
}


/*
 * CastingImpl type, this should be a subclass of in the future.
 */
NPY_NO_EXPORT PyTypeObject PyArrayCastingImpl_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.CastingImpl",
    .tp_basicsize = sizeof(CastingImpl),
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

