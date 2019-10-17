#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_pycompat.h"
#include "common.h"

#include "usertypes.h"
#include "ctors.h"

#include "common.h"
#include "npy_buffer.h"

#include "get_attr_string.h"
#include "mem_overlap.h"
#include "dtypemeta.h"
#include "convert_datatype.h"

/*
 * The casting to use for implicit assignment operations resulting from
 * in-place operations (like +=) and out= arguments. (Notice that this
 * variable is misnamed, but it's part of the public API so I'm not sure we
 * can just change it. Maybe someone should try and see if anyone notices.
 */
/*
 * In numpy 1.6 and earlier, this was NPY_UNSAFE_CASTING. In a future
 * release, it will become NPY_SAME_KIND_CASTING.  Right now, during the
 * transitional period, we continue to follow the NPY_UNSAFE_CASTING rules (to
 * avoid breaking people's code), but we also check for whether the cast would
 * be allowed under the NPY_SAME_KIND_CASTING rules, and if not we issue a
 * warning (that people's code will be broken in a future release.)
 */

NPY_NO_EXPORT NPY_CASTING NPY_DEFAULT_ASSIGN_CASTING = NPY_SAME_KIND_CASTING;


NPY_NO_EXPORT PyArray_Descr *
_array_find_python_scalar_type(PyObject *op)
{
    if (PyFloat_Check(op)) {
        return PyArray_DescrFromType(NPY_DOUBLE);
    }
    else if (PyComplex_Check(op)) {
        return PyArray_DescrFromType(NPY_CDOUBLE);
    }
    else if (PyInt_Check(op)) {
        /* bools are a subclass of int */
        if (PyBool_Check(op)) {
            return PyArray_DescrFromType(NPY_BOOL);
        }
        else {
            return  PyArray_DescrFromType(NPY_LONG);
        }
    }
    else if (PyLong_Check(op)) {
        /* check to see if integer can fit into a longlong or ulonglong
           and return that --- otherwise return object */
        if ((PyLong_AsLongLong(op) == -1) && PyErr_Occurred()) {
            PyErr_Clear();
        }
        else {
            return PyArray_DescrFromType(NPY_LONGLONG);
        }

        if ((PyLong_AsUnsignedLongLong(op) == (unsigned long long) -1)
            && PyErr_Occurred()){
            PyErr_Clear();
        }
        else {
            return PyArray_DescrFromType(NPY_ULONGLONG);
        }

        return PyArray_DescrFromType(NPY_OBJECT);
    }
    return NULL;
}


static PyArray_Descr *
_dtype_from_buffer_3118(PyObject *memoryview)
{
    PyArray_Descr *descr;
    Py_buffer *view = PyMemoryView_GET_BUFFER(memoryview);
    if (view->format != NULL) {
        descr = _descriptor_from_pep3118_format(view->format);
        if (descr == NULL) {
            return NULL;
        }
    }
    else {
        /* If no format is specified, just assume a byte array
         * TODO: void would make more sense here, as it wouldn't null
         *       terminate.
         */
        descr = PyArray_DescrNewFromType(NPY_STRING);
        descr->elsize = view->itemsize;
    }
    return descr;
}


static PyObject *
_array_from_buffer_3118(PyObject *memoryview)
{
    /* PEP 3118 */
    Py_buffer *view;
    PyArray_Descr *descr = NULL;
    PyObject *r = NULL;
    int nd, flags;
    Py_ssize_t d;
    npy_intp shape[NPY_MAXDIMS], strides[NPY_MAXDIMS];

    view = PyMemoryView_GET_BUFFER(memoryview);
    nd = view->ndim;
    descr = _dtype_from_buffer_3118(memoryview);

    if (descr == NULL) {
        return NULL;
    }

    /* Sanity check */
    if (descr->elsize != view->itemsize) {
        /* Ctypes has bugs in its PEP3118 implementation, which we need to
         * work around.
         *
         * bpo-10746
         * bpo-32780
         * bpo-32782
         *
         * Note that even if the above are fixed in master, we have to drop the
         * early patch versions of python to actually make use of the fixes.
         */
        if (!npy_ctypes_check(Py_TYPE(view->obj))) {
            /* This object has no excuse for a broken PEP3118 buffer */
            PyErr_Format(
                    PyExc_RuntimeError,
                   "Item size %zd for PEP 3118 buffer format "
                    "string %s does not match the dtype %c item size %d.",
                    view->itemsize, view->format, descr->type,
                    descr->elsize);
            Py_DECREF(descr);
            return NULL;
        }

        if (PyErr_Warn(
                    PyExc_RuntimeWarning,
                    "A builtin ctypes object gave a PEP3118 format "
                    "string that does not match its itemsize, so a "
                    "best-guess will be made of the data type. "
                    "Newer versions of python may behave correctly.") < 0) {
            Py_DECREF(descr);
            return NULL;
        }

        /* Thankfully, np.dtype(ctypes_type) works in most cases.
         * For an array input, this produces a dtype containing all the
         * dimensions, so the array is now 0d.
         */
        nd = 0;
        Py_DECREF(descr);
        descr = (PyArray_Descr *)PyObject_CallFunctionObjArgs(
                (PyObject *)&PyArrayDescr_Type, Py_TYPE(view->obj), NULL);
        if (descr == NULL) {
            return NULL;
        }
        if (descr->elsize != view->len) {
            PyErr_SetString(
                    PyExc_RuntimeError,
                    "For the given ctypes object, neither the item size "
                    "computed from the PEP 3118 buffer format nor from "
                    "converting the type to a np.dtype matched the actual "
                    "size. This is a bug both in python and numpy");
            Py_DECREF(descr);
            return NULL;
        }
    }

    if (view->shape != NULL) {
        int k;
        if (nd > NPY_MAXDIMS || nd < 0) {
            PyErr_Format(PyExc_RuntimeError,
                "PEP3118 dimensions do not satisfy 0 <= ndim <= NPY_MAXDIMS");
            goto fail;
        }
        for (k = 0; k < nd; ++k) {
            shape[k] = view->shape[k];
        }
        if (view->strides != NULL) {
            for (k = 0; k < nd; ++k) {
                strides[k] = view->strides[k];
            }
        }
        else {
            d = view->len;
            for (k = 0; k < nd; ++k) {
                if (view->shape[k] != 0) {
                    d /= view->shape[k];
                }
                strides[k] = d;
            }
        }
    }
    else {
        if (nd == 1) {
            shape[0] = view->len / view->itemsize;
            strides[0] = view->itemsize;
        }
        else if (nd > 1) {
            PyErr_SetString(PyExc_RuntimeError,
                           "ndim computed from the PEP 3118 buffer format "
                           "is greater than 1, but shape is NULL.");
            goto fail;
        }
    }

    flags = NPY_ARRAY_BEHAVED & (view->readonly ? ~NPY_ARRAY_WRITEABLE : ~0);
    r = PyArray_NewFromDescrAndBase(
            &PyArray_Type, descr,
            nd, shape, strides, view->buf,
            flags, NULL, memoryview);
    return r;


fail:
    Py_XDECREF(r);
    Py_XDECREF(descr);
    return NULL;

}


/*
 * Attempts to extract an array from an array-like object.
 *
 * array-like is defined as either
 *
 * * an object implementing the PEP 3118 buffer interface;
 * * an object with __array_struct__ or __array_interface__ attributes;
 * * an object with an __array__ function.
 *
 * Returns Py_NotImplemented if a given object is not array-like;
 * PyArrayObject* in case of success and NULL in case of failure.
 */
NPY_NO_EXPORT PyObject *
_array_from_array_like(PyObject *op, PyArray_Descr *requested_dtype,
                       npy_bool writeable, PyObject *context) {
    PyObject* tmp;

    /* If op supports the PEP 3118 buffer interface */
    if (!PyBytes_Check(op) && !PyUnicode_Check(op)) {
        PyObject *memoryview = PyMemoryView_FromObject(op);
        if (memoryview == NULL) {
            PyErr_Clear();
        }
        else {
            tmp = _array_from_buffer_3118(memoryview);
            Py_DECREF(memoryview);
            if (tmp == NULL) {
                return NULL;
            }

            if (writeable
                && PyArray_FailUnlessWriteable((PyArrayObject *) tmp, "PEP 3118 buffer") < 0) {
                Py_DECREF(tmp);
                return NULL;
            }

            return tmp;
        }
    }

    /* If op supports the __array_struct__ or __array_interface__ interface */
    tmp = PyArray_FromStructInterface(op);
    if (tmp == NULL) {
        return NULL;
    }
    if (tmp == Py_NotImplemented) {
        tmp = PyArray_FromInterface(op);
        if (tmp == NULL) {
            return NULL;
        }
    }

    /*
     * If op supplies the __array__ function.
     * The documentation says this should produce a copy, so
     * we skip this method if writeable is true, because the intent
     * of writeable is to modify the operand.
     * XXX: If the implementation is wrong, and/or if actual
     *      usage requires this behave differently,
     *      this should be changed!
     */
    if (!writeable && tmp == Py_NotImplemented) {
        tmp = PyArray_FromArrayAttr(op, requested_dtype, context);
        if (tmp == NULL) {
            return NULL;
        }
    }

    if (tmp != Py_NotImplemented) {
        if (writeable
            && PyArray_FailUnlessWriteable((PyArrayObject *) tmp,
                                           "array interface object") < 0) {
            Py_DECREF(tmp);
            return NULL;
        }
        return tmp;
    }

    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}


/* new reference */
NPY_NO_EXPORT PyArray_Descr *
_array_typedescr_fromstr(char *c_str)
{
    PyArray_Descr *descr = NULL;
    PyObject *stringobj = PyString_FromString(c_str);

    if (stringobj == NULL) {
        return NULL;
    }
    if (PyArray_DescrConverter(stringobj, &descr) != NPY_SUCCEED) {
        Py_DECREF(stringobj);
        return NULL;
    }
    Py_DECREF(stringobj);
    return descr;
}


NPY_NO_EXPORT char *
index2ptr(PyArrayObject *mp, npy_intp i)
{
    npy_intp dim0;

    if (PyArray_NDIM(mp) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
        return NULL;
    }
    dim0 = PyArray_DIMS(mp)[0];
    if (check_and_adjust_index(&i, dim0, 0, NULL) < 0)
        return NULL;
    if (i == 0) {
        return PyArray_DATA(mp);
    }
    return PyArray_BYTES(mp)+i*PyArray_STRIDES(mp)[0];
}

NPY_NO_EXPORT int
_zerofill(PyArrayObject *ret)
{
    if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
        PyObject *zero = PyInt_FromLong(0);
        PyArray_FillObjectArray(ret, zero);
        Py_DECREF(zero);
        if (PyErr_Occurred()) {
            Py_DECREF(ret);
            return -1;
        }
    }
    else {
        npy_intp n = PyArray_NBYTES(ret);
        memset(PyArray_DATA(ret), 0, n);
    }
    return 0;
}

NPY_NO_EXPORT npy_bool
_IsWriteable(PyArrayObject *ap)
{
    PyObject *base = PyArray_BASE(ap);
#if defined(NPY_PY3K)
    Py_buffer view;
#else
    void *dummy;
    Py_ssize_t n;
#endif

    /*
     * C-data wrapping arrays may not own their data while not having a base;
     * WRITEBACKIFCOPY arrays have a base, but do own their data.
     */
    if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
        /*
         * This is somewhat unsafe for directly wrapped non-writable C-arrays,
         * which do not know whether the memory area is writable or not and
         * do not own their data (but have no base).
         * It would be better if this returned PyArray_ISWRITEABLE(ap).
         * Since it is hard to deprecate, this is deprecated only on the Python
         * side, but not on in PyArray_UpdateFlags.
         */
        return NPY_TRUE;
    }

    /*
     * Get to the final base object.
     * If it is a writeable array, then return True if we can
     * find an array object or a writeable buffer object as
     * the final base object.
     */
    while (PyArray_Check(base)) {
        ap = (PyArrayObject *)base;
        base = PyArray_BASE(ap);

        if (PyArray_ISWRITEABLE(ap)) {
            /*
             * If any base is writeable, it must be OK to switch, note that
             * bases are typically collapsed to always point to the most
             * general one.
             */
            return NPY_TRUE;
        }

        if (base == NULL || PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA)) {
            /* there is no further base to test the writeable flag for */
            return NPY_FALSE;
        }
        assert(!PyArray_CHKFLAGS(ap, NPY_ARRAY_OWNDATA));
    }

#if defined(NPY_PY3K)
    if (PyObject_GetBuffer(base, &view, PyBUF_WRITABLE|PyBUF_SIMPLE) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
    PyBuffer_Release(&view);
    /*
     * The first call to PyObject_GetBuffer stores a reference to a struct
     * _buffer_info_t (from buffer.c, with format, ndim, strides and shape) in
     * a static dictionary, with id(base) as the key. Usually we release it
     * after the call to PyBuffer_Release, via a call to
     * _dealloc_cached_buffer_info, but in this case leave it in the cache to
     * speed up future calls to _IsWriteable.
     */
#else
    if (PyObject_AsWriteBuffer(base, &dummy, &n) < 0) {
        PyErr_Clear();
        return NPY_FALSE;
    }
#endif
    return NPY_TRUE;
}


/**
 * Convert an array shape to a string such as "(1, 2)".
 *
 * @param Dimensionality of the shape
 * @param npy_intp pointer to shape array
 * @param String to append after the shape `(1, 2)%s`.
 *
 * @return Python unicode string
 */
NPY_NO_EXPORT PyObject *
convert_shape_to_string(npy_intp n, npy_intp *vals, char *ending)
{
    npy_intp i;
    PyObject *ret, *tmp;

    /*
     * Negative dimension indicates "newaxis", which can
     * be discarded for printing if it's a leading dimension.
     * Find the first non-"newaxis" dimension.
     */
    for (i = 0; i < n && vals[i] < 0; i++);

    if (i == n) {
        return PyUString_FromFormat("()%s", ending);
    }
    else {
        ret = PyUString_FromFormat("(%" NPY_INTP_FMT, vals[i++]);
        if (ret == NULL) {
            return NULL;
        }
    }

    for (; i < n; ++i) {
        if (vals[i] < 0) {
            tmp = PyUString_FromString(",newaxis");
        }
        else {
            tmp = PyUString_FromFormat(",%" NPY_INTP_FMT, vals[i]);
        }
        if (tmp == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        PyUString_ConcatAndDel(&ret, tmp);
        if (ret == NULL) {
            return NULL;
        }
    }

    if (i == 1) {
        tmp = PyUString_FromFormat(",)%s", ending);
    }
    else {
        tmp = PyUString_FromFormat(")%s", ending);
    }
    PyUString_ConcatAndDel(&ret, tmp);
    return ret;
}


NPY_NO_EXPORT void
dot_alignment_error(PyArrayObject *a, int i, PyArrayObject *b, int j)
{
    PyObject *errmsg = NULL, *format = NULL, *fmt_args = NULL,
             *i_obj = NULL, *j_obj = NULL,
             *shape1 = NULL, *shape2 = NULL,
             *shape1_i = NULL, *shape2_j = NULL;

    format = PyUString_FromString("shapes %s and %s not aligned:"
                                  " %d (dim %d) != %d (dim %d)");

    shape1 = convert_shape_to_string(PyArray_NDIM(a), PyArray_DIMS(a), "");
    shape2 = convert_shape_to_string(PyArray_NDIM(b), PyArray_DIMS(b), "");

    i_obj = PyLong_FromLong(i);
    j_obj = PyLong_FromLong(j);

    shape1_i = PyLong_FromSsize_t(PyArray_DIM(a, i));
    shape2_j = PyLong_FromSsize_t(PyArray_DIM(b, j));

    if (!format || !shape1 || !shape2 || !i_obj || !j_obj ||
            !shape1_i || !shape2_j) {
        goto end;
    }

    fmt_args = PyTuple_Pack(6, shape1, shape2,
                            shape1_i, i_obj, shape2_j, j_obj);
    if (fmt_args == NULL) {
        goto end;
    }

    errmsg = PyUString_Format(format, fmt_args);
    if (errmsg != NULL) {
        PyErr_SetObject(PyExc_ValueError, errmsg);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "shapes are not aligned");
    }

end:
    Py_XDECREF(errmsg);
    Py_XDECREF(fmt_args);
    Py_XDECREF(format);
    Py_XDECREF(i_obj);
    Py_XDECREF(j_obj);
    Py_XDECREF(shape1);
    Py_XDECREF(shape2);
    Py_XDECREF(shape1_i);
    Py_XDECREF(shape2_j);
}

/**
 * unpack tuple of dtype->fields (descr, offset, title[not-needed])
 *
 * @param "value" should be the tuple.
 *
 * @return "descr" will be set to the field's dtype
 * @return "offset" will be set to the field's offset
 *
 * returns -1 on failure, 0 on success.
 */
NPY_NO_EXPORT int
_unpack_field(PyObject *value, PyArray_Descr **descr, npy_intp *offset)
{
    PyObject * off;
    if (PyTuple_GET_SIZE(value) < 2) {
        return -1;
    }
    *descr = (PyArray_Descr *)PyTuple_GET_ITEM(value, 0);
    off  = PyTuple_GET_ITEM(value, 1);

    if (PyInt_Check(off)) {
        *offset = PyInt_AsSsize_t(off);
    }
    else if (PyLong_Check(off)) {
        *offset = PyLong_AsSsize_t(off);
    }
    else {
        PyErr_SetString(PyExc_IndexError, "can't convert offset");
        return -1;
    }

    return 0;
}

/*
 * check whether arrays with datatype dtype might have object fields. This will
 * only happen for structured dtypes (which may have hidden objects even if the
 * HASOBJECT flag is false), object dtypes, or subarray dtypes whose base type
 * is either of these.
 */
NPY_NO_EXPORT int
_may_have_objects(PyArray_Descr *dtype)
{
    PyArray_Descr *base = dtype;
    if (PyDataType_HASSUBARRAY(dtype)) {
        base = dtype->subarray->base;
    }

    return (PyDataType_HASFIELDS(base) ||
            PyDataType_FLAGCHK(base, NPY_ITEM_HASOBJECT) );
}

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
NPY_NO_EXPORT PyArrayObject *
new_array_for_sum(PyArrayObject *ap1, PyArrayObject *ap2, PyArrayObject* out,
                  int nd, npy_intp dimensions[], int typenum, PyArrayObject **result)
{
    PyArrayObject *out_buf;

    if (out) {
        int d;

        /* verify that out is usable */
        if (PyArray_NDIM(out) != nd ||
            PyArray_TYPE(out) != typenum ||
            !PyArray_ISCARRAY(out)) {
            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return 0;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                return 0;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = (PyArrayObject *)PyArray_NewLikeArray(out, NPY_CORDER,
                                                            NULL, 0);
            if (out_buf == NULL) {
                return NULL;
            }

            /* set copy-back */
            Py_INCREF(out);
            if (PyArray_SetWritebackIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                Py_DECREF(out_buf);
                return NULL;
            }
        }
        else {
            Py_INCREF(out);
            out_buf = out;
        }

        if (result) {
            Py_INCREF(out);
            *result = out;
        }

        return out_buf;
    }
    else {
        PyTypeObject *subtype;
        double prior1, prior2;
        /*
         * Need to choose an output array that can hold a sum
         * -- use priority to determine which subtype.
         */
        if (Py_TYPE(ap2) != Py_TYPE(ap1)) {
            prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
            prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
            subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
        }
        else {
            prior1 = prior2 = 0.0;
            subtype = Py_TYPE(ap1);
        }

        out_buf = (PyArrayObject *)PyArray_New(subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0,
                                               (PyObject *)
                                               (prior2 > prior1 ? ap2 : ap1));

        if (out_buf != NULL && result) {
            Py_INCREF(out_buf);
            *result = out_buf;
        }

        return out_buf;
    }
}


static int
update_shape(int curr_ndim, int *max_ndim,
             npy_intp out_shape[NPY_MAXDIMS], int new_ndim,
             const npy_intp new_shape[NPY_MAXDIMS], npy_bool sequence)
{
    int success = 0;  /* unsuccessful if array is ragged */
    if (curr_ndim + new_ndim > *max_ndim) {
        success = -1;
        /* Only update check as many dims as possible, max_ndim is unchanged */
        new_ndim = *max_ndim - curr_ndim;
    }
    else if (!sequence && (*max_ndim != curr_ndim + new_ndim)) {
        /*
         * Sequences do not update max_ndim, otherwise shrink and check.
         * This is depth first, so if it is already set, `out_shape` is filled.
         */
        *max_ndim = curr_ndim + new_ndim;
        /* If a shape was already set, this is also ragged */
        if (out_shape[*max_ndim] >= 0) {
            success = -1;
        }
    }
    for (int i = 0; i < new_ndim; i++) {
        npy_intp curr_dim = out_shape[curr_ndim + i];
        npy_intp new_dim = new_shape[i];

        if (curr_dim == -1) {
            out_shape[curr_ndim + i] = new_dim;
        }
        else if (new_dim != curr_dim) {
            /* The array is ragged, and this dimension is unusable already */
            success = -1;
            if (!sequence) {
                /* Remove dimensions that we cannot use: */
                *max_ndim -= new_ndim + i;
            }
            else {
                assert(i == 0);
                /* max_ndim is usually not updated for sequences, so set now: */
                *max_ndim = curr_ndim;
            }
            break;
        }
    }
    return success;
}


/*
 * This cache is necessary for the simple case of an array input mostly.
 * Since that is the main reason, this may be removed if the single array
 * case is handled specifically up-front.
 * This may be better as a different caching mechanism...
 */
#define COERCION_CACHE_SIZE 5
static coercion_cache_obj *global_coercion_cache[COERCION_CACHE_SIZE] = {NULL};

NPY_NO_EXPORT int
npy_new_coercion_cache(
        PyObject *converted_obj, PyObject *arr_or_sequence, npy_bool sequence,
        coercion_cache_obj ***next_ptr)
{
    coercion_cache_obj *cache = NULL;
    for (int i = 0; i < COERCION_CACHE_SIZE; i++) {
        if (global_coercion_cache[i] != NULL) {
            cache = global_coercion_cache[i];
            global_coercion_cache[i] = NULL;
            break;
        }
    }
    if (cache == NULL) {
        cache = PyArray_malloc(sizeof(coercion_cache_obj));
        if (cache == NULL) {
            PyErr_NoMemory();
            return -1;
        }
    }
    cache->converted_obj = converted_obj;
    Py_INCREF(arr_or_sequence);
    cache->arr_or_sequence = arr_or_sequence;
    cache->sequence = sequence;
    cache->next = NULL;
    **next_ptr = cache;
    *next_ptr = &(cache->next);
    return 0;
}


NPY_NO_EXPORT void npy_free_coercion_cache(coercion_cache_obj *next) {
    /* We only need to check from the last used cache pos */
    int cache_pos = 0;
    while (next != NULL) {
        coercion_cache_obj *current = next;
        next = current->next;

        Py_DECREF(current->arr_or_sequence);
        for (; cache_pos < COERCION_CACHE_SIZE; cache_pos++) {
            if (global_coercion_cache[cache_pos] == NULL) {
                global_coercion_cache[cache_pos] = current;
                break;
            }
        }
        if (cache_pos == COERCION_CACHE_SIZE) {
            PyArray_free(current);
        }
    }
}

#undef COERCION_CACHE_SIZE

/*
 * Recursive helper of the `PyArray_DiscoverDTypeFromObject` function.
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeFromObjectRecursive(
        PyObject *obj, int max_dims, int curr_dims,
        PyArray_DTypeMeta **out_dtype, npy_intp out_shape[NPY_MAXDIMS],
        npy_bool use_minimal, npy_bool *single_or_no_element,
        /* These two are solely for the __array__ attribute */
        PyArray_Descr *requested_dtype,
        PyObject *context,
        // TODO: Hacks to support legay behaviour (at least second one)
        npy_bool stop_at_tuple, npy_bool string_is_sequence,
        PyTypeObject *prev_type, PyArray_DTypeMeta *prev_dtype,
        coercion_cache_obj ***coercion_cache_tail_ptr)
{
    PyArray_DTypeMeta *dtype = NULL;
    PyArray_Descr *descriptor = NULL;

    /* obj is a string and we have a "c" dtype, so make the string a sequence */
    // TODO: slow, but do this check first (it should never be used)
    if (string_is_sequence) {
        /* Do not bother to promote, it was already defined as a char. */
        if (PyString_Check(obj) || PyUnicode_Check(obj)) {
            /* Of course we can only do that if there is more than one char */
            if (PySequence_Length(obj) != 1) {
                goto force_sequence;
            }
        }
    }

    if (prev_type == Py_TYPE(obj)) {
        /* super-fast check for the common case of homogeneous sequences */
        dtype = prev_dtype;
    }
    else {
        dtype = (PyArray_DTypeMeta *)PyDict_GetItem(
                PyArrayDTypeMeta_associated_types, (PyObject *) Py_TYPE(obj));
    }
    if (dtype != NULL) {
        /* Cache that we have discovered the dtype class: */
        prev_type = Py_TYPE(obj);
        prev_dtype = dtype;

        if (update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
            goto ragged_array;
        }

        // TODO: At least if the below does not return an abstract dtype
        //       we could cache that directly (although it is unclear if there
        //       actually is a clear use-case for that. Which may mean the
        //       function should be just called pybject and the other removed.
        //       (The only reason for it would be if we do not have the
        //       associated types dictionary above?)
        if (dtype->abstract) {
            if (dtype->dt_slots->requires_pyobject_for_discovery) {
                dtype = dtype->dt_slots->discover_dtype_from_pytype(
                        dtype, obj, use_minimal);
            }
            else {
                dtype = dtype->dt_slots->discover_dtype_from_pytype(
                        dtype, (PyObject *)Py_TYPE(obj), use_minimal);
            }
            if (dtype == NULL) {
                goto fail;
            }
        }
        else {
            Py_INCREF(dtype);
        }
        goto promote_types;
    }

    /* obj is a Tuple, but tuples aren't expanded */
    if (stop_at_tuple && PyTuple_Check(obj)) {
        /* Do not bother to promote, dtype instance must have been passed in */
        if (update_shape(
                curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
            /* But do update, if there this is a ragged case */
            goto ragged_array;
        }
        // TODO: Should be able to just do nothing with dtype, but for now to
        //       avoid bugs down here, set to OBJECT.
        descriptor = PyArray_DescrFromType(NPY_OBJECT);
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        Py_DECREF(descriptor);
        goto promote_types;
    }

    /* Check if it's an ndarray */
    if (PyArray_Check(obj)) {
        if (update_shape(curr_dims, &max_dims, out_shape,
                         PyArray_NDIM(obj),
                         PyArray_SHAPE((PyArrayObject *)obj), NPY_FALSE) < 0) {
            goto ragged_array;
        }
        descriptor = PyArray_DESCR((PyArrayObject *)obj);
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        /* We must cache it for dtype discovery currently (it hardly hurts) */
        if (npy_new_coercion_cache(obj, obj, 0, coercion_cache_tail_ptr) < 0) {
            goto fail;
        }
        goto promote_types;
    }

    /* Check if it's a NumPy scalar */
    // TODO: Should be found before, unless a subclass, which is no good?
    if (PyArray_IsScalar(obj, Generic)) {
        if (update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
            goto ragged_array;
        }
        descriptor = PyArray_DescrFromScalar(obj);
        if (descriptor == NULL) {
            goto fail;
        }
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        Py_DECREF(descriptor);
        prev_type = Py_TYPE(obj);
        prev_dtype = dtype;
        goto promote_types;
    }

    // TODO: We need to do this, since we detect subclasses currently,
    //       and not just exact matches! (Make sure we have tests for this!)
    // TODO: Need to fix to use the abstract dtypes here (OTOH, if we deprecate
    //       it, it does not matter maybe, since it is only incorrect after
    //       the outer change/deprecation of fixing scalar handling in ufuncs).
    // TODO: Deprecate!
    /* Check if it's a Python scalar */
    descriptor = _array_find_python_scalar_type(obj);
    if (descriptor != NULL) {
        if (update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE) < 0) {
            goto ragged_array;
        }
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        Py_DECREF(descriptor);
        goto promote_types;
    }
    /* Check if it's an ASCII string */
    {
        // TODO: These must remain as fallbacks for subclasses (Deprecate)
        if (PyBytes_Check(obj)) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "this is not yet implemented, and should not really happen much.");
            goto fail;
        }

        /* Check if it's a Unicode string */
        if (PyUnicode_Check(obj)) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "this is not yet implemented, and should not really happen much.");
            goto fail;
        }
    }

    {
        PyObject *tmp = _array_from_array_like(
                obj,  requested_dtype,0,
                /* context is UFunc call info; not passed to nested objs */
                curr_dims == 0 ? context : NULL);
        if (tmp == NULL) {
            goto fail;
        }
        else if (tmp != Py_NotImplemented) {
            if (update_shape(curr_dims, &max_dims, out_shape,
                             PyArray_NDIM(tmp),
                             PyArray_SHAPE((PyArrayObject *)tmp), NPY_FALSE) < 0) {
                Py_DECREF(tmp);
                goto ragged_array;
            }
            descriptor = PyArray_DESCR(tmp);
            if (npy_new_coercion_cache(obj, tmp, 0, coercion_cache_tail_ptr) < 0) {
                Py_DECREF(tmp);
                goto fail;
            }
            Py_DECREF(tmp);
            dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
            Py_INCREF(dtype);
            goto promote_types;
        }
        Py_DECREF(tmp);
    }


force_sequence:
    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, and obj isn't a sequence-like object, the output
     * dtype should be either OBJECT or a user-defined type.
     *
     * Note that some libraries define sequence-like classes but want them to
     * be treated as objects, and they expect numpy to treat it as an object if
     * __len__ is not defined.
     */
    if (!PySequence_Check(obj) || PySequence_Size(obj) < 0) {
        /* clear any PySequence_Size error which corrupts further calls */
        PyErr_Clear();

        /* This branch always leads to a ragged array */
        update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE);
        goto ragged_array;
    }

    /*
     * The C-API recommends calling PySequence_Fast before any of the other
     * PySequence_Fast* functions. This is required for PyPy
     */
    PyObject *seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        /* Specifically do not fail on things that look like a dictionary */
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            update_shape(curr_dims, &max_dims, out_shape, 0, NULL, NPY_FALSE);
            goto ragged_array;
        }
        goto fail;
    }
    if (npy_new_coercion_cache(obj, seq, 1, coercion_cache_tail_ptr) < 0) {
        Py_DECREF(seq);
        goto fail;
    }

    npy_intp size = PySequence_Fast_GET_SIZE(seq);
    PyObject **objects = PySequence_Fast_ITEMS(seq);

    if (update_shape(curr_dims, &max_dims,
                     out_shape, 1, &size, NPY_TRUE) < 0) {
        /* But do update, if there this is a ragged case */
        Py_DECREF(seq);
        goto ragged_array;
    }
    if (size == 0) {
        Py_DECREF(seq);
        /* If the sequence is empty, we have to assume thats it... */
        return curr_dims+1;
    }

    /* Recursive call for each sequence item */
    for (Py_ssize_t i = 0; i < size; ++i) {
        max_dims = PyArray_DiscoverDTypeFromObjectRecursive(
                objects[i], max_dims, curr_dims + 1,
                out_dtype, out_shape, use_minimal,
                single_or_no_element, requested_dtype, context,
                stop_at_tuple, string_is_sequence,
                prev_type, prev_dtype, coercion_cache_tail_ptr);
        // NOTE: If there is a ragged array found (NPY_OBJECT) could break
        if (max_dims < 0) {
            Py_DECREF(seq);
            goto fail;
        }
    }
    Py_DECREF(seq);
    return max_dims;

ragged_array:
    // TODO: We may want to add a deprecation warning in this path,
    //       in generally, object should probably never happen without
    //       the user asking for it specifically.
    // NOTE: Users can probably supply max_dims and out_dtype.
    if (*out_dtype != NULL) {
        *single_or_no_element = 0;
    }
    if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
        Py_XDECREF(*out_dtype);
        descriptor = PyArray_DescrFromType(NPY_OBJECT);
        *out_dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(*out_dtype);
        Py_DECREF(descriptor);
    }
    return max_dims;

promote_types:
    if (*out_dtype == NULL) {
        *out_dtype = dtype;
        return max_dims;
    }
    *single_or_no_element = 0;
    Py_SETREF(*out_dtype, PyArray_PromoteDTypes(*out_dtype, dtype));
    Py_DECREF(dtype);
    if (*out_dtype == NULL) {
        PyErr_Clear();
        /*
         * Fallback to object, at this point this is OK, later on an error
         * should likely be raised if the user did not provide a DType.
         */
        descriptor = PyArray_DescrFromType(NPY_OBJECT);
        *out_dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(*out_dtype);
        Py_DECREF(descriptor);
    }
    return max_dims;

fail:
    Py_XDECREF(*out_dtype);
    *out_dtype = NULL;
    return -1;
}


/*
 * Internal helper to find the correct DType (class). Must be called with
 * `curr_dim = 0`. Returns the maximum reached depth and a negative number
 * on failure. `out_dtype` is NULL on error, otherwise a reference to a DType
 * class.
 * The function fills in the resulting shape in `out_shape`.
 * The caller may get an abstract dtype returned, at which point it is may
 * be necessary to convert it.
 * If `use_minimal` is set, certain abstract dtypes may return a different
 * dtype. (e.g. a python int is always long, unsigned long or object and not
 * and abstract dtype describing the minimal possible dtype to hold the data).
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeFromObject(
        PyObject *obj, int max_dims,
        PyArray_DTypeMeta **out_dtype, npy_intp out_shape[NPY_MAXDIMS],
        npy_bool use_minimal, coercion_cache_obj **coercion_cache,
        npy_bool *single_or_no_element,
        /* These two are solely for the __array__ attribute */
        PyArray_Descr *requested_dtype,
        PyObject *context,
        // TODO: Hacks to support legay behaviour (at least second one)
        npy_bool stop_at_tuple, npy_bool string_is_sequence)
{
    // TODO: Should likely add an exact array match fast path here and
    //       possibly even a scalar fast path.
    //       Similarly the "writeable" path of GetArrayParamsFromObj may be
    //       interesting outside of the recursive path.

    PyTypeObject *prev_type = NULL;
    PyArray_DTypeMeta *prev_dtype = NULL;

    // TODO: May want to put a lot of this into a single struct
    //       stack allocated by the caller instead of many parameters?
    /* Do housekeeping for the initial call in the recursion: */
    *coercion_cache = NULL;
    *single_or_no_element = 1;

    /* initialize shape for shape discovery */
    for (int i = 0; i < max_dims; i++) {
        out_shape[i] = -1;
    }

    return PyArray_DiscoverDTypeFromObjectRecursive(
            obj, max_dims, 0, out_dtype, out_shape,
            use_minimal, single_or_no_element,
            requested_dtype, context,
            stop_at_tuple, string_is_sequence,
            prev_type, prev_dtype, &coercion_cache);
}


static int
PyArray_DiscoverDescriptorFromObjectRecursive(
        PyObject *obj,
        PyArray_Descr **out_descr, coercion_cache_obj **coercion_cache,
        PyArray_DTypeMeta *dtype,
        PyTypeObject **last_dtype, CastingImpl **last_castingimpl)
{
    coercion_cache_obj *cache = *coercion_cache;
    PyArray_Descr *descr;
    if (cache != NULL && obj == cache->converted_obj) {
        /* Advance the coercion_cache linked list. This one is now used. */
        *coercion_cache = cache->next;

        if (cache->sequence) {
            PyObject *seq = cache->arr_or_sequence;
            npy_intp size = PySequence_Fast_GET_SIZE(seq);
            PyObject **objects = PySequence_Fast_ITEMS(seq);
            for (npy_intp i = 0; i < size; i++) {
                int res = PyArray_DiscoverDescriptorFromObjectRecursive(
                        objects[i], out_descr, coercion_cache,
                        dtype, last_dtype, last_castingimpl);
                if (res < 0) {
                    return -1;
                }
            }
            /* No need to find the common descriptor instance */
            return 0;
        }
        else {
            descr = PyArray_DESCR(cache->arr_or_sequence);
            Py_INCREF(descr);
            if (Py_TYPE(descr) != (PyTypeObject *) dtype) {
                /*
                 * If this is not an instance of the correct dtype class,
                 * need to use the CastingImpl to find the correct descriptor.
                 */
                CastingImpl *casting_impl;
                if (Py_TYPE(descr) == *last_dtype) {
                    casting_impl = *last_castingimpl;
                    Py_INCREF(casting_impl);
                }
                else {
                    casting_impl = get_casting_impl(
                            (PyArray_DTypeMeta *) Py_TYPE(descr), dtype,
                            NPY_UNSAFE_CASTING);
                    if (casting_impl == NULL) {
                        // TODO: Should use a goto fail probably.
                        Py_DECREF(descr);
                        Py_XDECREF(*out_descr);
                        *out_descr = NULL;
                        return -1;
                    }
                    /* replace the cached casting impl */
                    Py_INCREF(Py_TYPE(descr));
                    Py_XSETREF(*last_dtype, Py_TYPE(descr));
                    Py_INCREF(casting_impl);
                    Py_XSETREF(*last_castingimpl, casting_impl);
                }

                PyArray_Descr * in_descrs[2] = {descr, NULL};
                PyArray_Descr * out_descrs[2];
                int success = casting_impl->adjust_descriptors(
                        casting_impl, in_descrs, out_descrs, NPY_UNSAFE_CASTING);
                if (success < 0) {
                    Py_DECREF(descr);
                    Py_XDECREF(*out_descr);
                    *out_descr = NULL;
                    return -1;
                }
                Py_DECREF(descr);
                descr = out_descrs[1];
                Py_DECREF(out_descrs[0]);
            }
        }
    }
    else {
        descr = dtype->dt_slots->discover_descr_from_pyobject(dtype, obj);
        if (descr == NULL) {
            Py_XDECREF(*out_descr);
            *out_descr = NULL;
            return -1;
        }
    }
    if (*out_descr == NULL) {
        *out_descr = descr;
        return 0;
    }
    /* We still need to find the common dtype/descriptor instance */
    PyArray_Descr *common_descr = dtype->dt_slots->common_instance(dtype,
            *out_descr, descr);
    Py_DECREF(descr);
    Py_DECREF(*out_descr);
    if (common_descr == NULL) {
        *out_descr = NULL;
        return -1;
    }
    *out_descr = common_descr;
    return 0;
}

NPY_NO_EXPORT int
PyArray_DiscoverDescriptorFromObject(
        PyObject *obj,
        PyArray_Descr **out_descr, coercion_cache_obj **coercion_cache,
        npy_bool single_or_no_element, PyArray_DTypeMeta *dtype) {
    *out_descr = NULL;
    if (dtype == NULL) {
        /* An empty sequence, let the user deal with it. */
        assert(single_or_no_element);
        return 0;
    }
    if (!dtype->flexible) {
        /* This is fairly boring (usually) */
        if (!single_or_no_element) {
            *out_descr = dtype->dt_slots->default_descr(dtype);
            if (*out_descr == NULL) {
                return -1;
            }
            return 0;
        }
        /* There is a single element somewhere, it may have a descr. */
    }
    PyTypeObject *last_dtype = NULL;
    CastingImpl *last_castingimpl = NULL;
    /* Copy pointer, so recursive function can advance/share it */
    coercion_cache_obj *coercion_cache_copy = *coercion_cache;
    int res = PyArray_DiscoverDescriptorFromObjectRecursive(
            obj, out_descr, &coercion_cache_copy, dtype,
            &last_dtype, &last_castingimpl);
    /*
     * It is possible that the above did not return a result
     * (when it is empty), in that case, probably are seeing a string
     * of length zero or so, but...
     */
    Py_XDECREF(last_dtype);
    Py_XDECREF(last_castingimpl);
    if (res < 0) {
       return res;
    }
    if (*out_descr == NULL) {
        assert(single_or_no_element);
        /* Try to get the default one... */
        *out_descr = dtype->dt_slots->default_descr(dtype);
        if (*out_descr == NULL) {
            return -1;
        }
    }
    return 0;
}
