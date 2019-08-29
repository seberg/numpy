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
        return NPY_FAIL;
    }
    out_descrs[1] = self->to_dtype->dt_slots->default_descr(self->to_dtype);
    if (out_descrs[1] == NULL) {
        Py_DECREF(out_descrs[0]);
        return NPY_FAIL;
    }

    Py_INCREF(in_descrs[0]);
    out_descrs[0] = in_descrs[0];

    return NPY_SUCCEED;
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
    
    return NPY_SUCCEED;
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
        int res = PyArray_CanCastTypeTo(in_descrs[0], in_descrs[1], casting);
        if (res) {
            // TODO: For testing, could not do the byteswapping here!
            Py_INCREF(in_descrs[0]);
            out_descrs[0] = in_descrs[0];
            Py_INCREF(in_descrs[1]);
            out_descrs[1] = in_descrs[1];
            return NPY_SUCCEED;
        }
        else {
            return NPY_FAIL;
        }
    }
    else {
        /*
         * We need to adapt the dtype, which means pomotion in old terms.
         */
        // This function wraps existing dtypes, user dtypes are never flexible
        // so this must be string, bytes, datetime, or timedelta.
        int from_type_num = in_descrs[0]->type_num;
        int to_type_num = in_descrs[0]->type_num;
        int valid;

        /* The legacy datatype should be string or unicode. */
        if (to_type_num == NPY_STRING || to_type_num == NPY_UNICODE) {
            PyArray_Descr *tmp = PyArray_DescrNewFromType(to_type_num);
            if (tmp == NULL) {
                return NPY_FAIL;
            }
            // TODO: For scalar objects we may have special logic here! ugg!
            out_descrs[1] = PyArray_AdaptFlexibleDType(NULL, in_descrs[1], tmp);
            Py_DECREF(tmp);
            if (out_descrs[1] == NULL) {
                return NPY_FAIL;
            }
            Py_INCREF(in_descrs[0]);
            out_descrs[0] = in_descrs[0];
        }
        // TODO: for Datetimes without a unit there may be corner cases...
        PyErr_SetString(PyExc_TypeError,
            "cannot discover correct legacy output dtype. "
            "Cast requires a specific output instance to work.");
        return NPY_FAIL;
    }
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
    CastingImpl *castingimpl = NULL;

    castingimpl = (CastingImpl *)PyArray_malloc(sizeof(CastingImpl));
    if (castingimpl == NULL) {
        return NULL;
    }
    memset(castingimpl, 0, sizeof(CastingImpl));
    // TODO: Can Init fail? This is likely not the nicest way anyway?
    PyObject_Init((PyObject *)castingimpl, &PyArrayCastingImpl_Type);

    castingimpl->from_dtype = from_dtype;
    castingimpl->to_dtype = to_dtype;

    if (from_dtype == to_dtype) {
        /*
         * We have to cast within a dtype, which needs to handle full alignment
         * and byte swapping.
         */
         castingimpl->adjust_descriptors = adjust_two_descriptors_within_dtype;
    }
    else {
        /*
         * Casting is between two types, so that it is allowed to be simple,
         * i.e. the casting functions do not necessarily need byte swapping.
         */
         if (from_dtype->flexible || to_dtype->flexible) {
            castingimpl->adjust_descriptors = adjust_two_descriptors_flexible;
         }
         else {
            castingimpl->adjust_descriptors = adjust_two_descriptors_trivial;
         }
    }
    
    return (PyObject *)castingimpl;
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

