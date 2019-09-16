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


/*
 * These constants are used to signal that the recursive dtype determination in
 * PyArray_DTypeFromObject encountered a string type, and that the recursive
 * search must be restarted so that string representation lengths can be
 * computed for all scalar types.
 */
#define RETRY_WITH_STRING 1
#define RETRY_WITH_UNICODE 2

/*
 * Recursively examines the object to determine an appropriate dtype
 * to use for converting to an ndarray.
 *
 * 'obj' is the object to be converted to an ndarray.
 *
 * 'maxdims' is the maximum recursion depth.
 *
 * 'out_dtype' should be either NULL or a minimal starting dtype when
 * the function is called. It is updated with the results of type
 * promotion. This dtype does not get updated when processing NA objects.
 * This is reset to NULL on failure.
 *
 * Returns 0 on success, -1 on failure.
 */
 NPY_NO_EXPORT int
PyArray_DTypeFromObject(PyObject *obj, int maxdims, PyArray_Descr **out_dtype)
{
    int res;

    res = PyArray_DTypeFromObjectHelper(obj, maxdims, out_dtype, 0);
    if (res == RETRY_WITH_STRING) {
        res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                            out_dtype, NPY_STRING);
        if (res == RETRY_WITH_UNICODE) {
            res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                                out_dtype, NPY_UNICODE);
        }
    }
    else if (res == RETRY_WITH_UNICODE) {
        res = PyArray_DTypeFromObjectHelper(obj, maxdims,
                                            out_dtype, NPY_UNICODE);
    }
    return res;
}

NPY_NO_EXPORT int
PyArray_DTypeFromObjectHelper(PyObject *obj, int maxdims,
                              PyArray_Descr **out_dtype, int string_type)
{
    int i, size;
    PyArray_Descr *descriptor = NULL;
    PyObject *ip;
    Py_buffer buffer_view;
    /* types for sequence handling */
    PyObject ** objects;
    PyObject * seq;
    PyTypeObject * common_type;

    /* Check if it's an ndarray */
    if (PyArray_Check(obj)) {
        descriptor = PyArray_DESCR((PyArrayObject *)obj);
        Py_INCREF(descriptor);
        goto promote_types;
    }

    /* See if it's a python None */
    if (obj == Py_None) {
        descriptor = PyArray_DescrFromType(NPY_OBJECT);
        if (descriptor == NULL) {
            goto fail;
        }
        goto promote_types;
    }
    /* Check if it's a NumPy scalar */
    else if (PyArray_IsScalar(obj, Generic)) {
        if (!string_type) {
            descriptor = PyArray_DescrFromScalar(obj);
            if (descriptor == NULL) {
                goto fail;
            }
        }
        else {
            int itemsize;
            PyObject *temp;

            if (string_type == NPY_STRING) {
                if ((temp = PyObject_Str(obj)) == NULL) {
                    goto fail;
                }
#if defined(NPY_PY3K)
    #if PY_VERSION_HEX >= 0x03030000
                itemsize = PyUnicode_GetLength(temp);
    #else
                itemsize = PyUnicode_GET_SIZE(temp);
    #endif
#else
                itemsize = PyString_GET_SIZE(temp);
#endif
            }
            else if (string_type == NPY_UNICODE) {
#if defined(NPY_PY3K)
                if ((temp = PyObject_Str(obj)) == NULL) {
#else
                if ((temp = PyObject_Unicode(obj)) == NULL) {
#endif
                    goto fail;
                }
                itemsize = PyUnicode_GET_DATA_SIZE(temp);
#ifndef Py_UNICODE_WIDE
                itemsize <<= 1;
#endif
            }
            else {
                goto fail;
            }
            Py_DECREF(temp);
            if (*out_dtype != NULL &&
                    (*out_dtype)->type_num == string_type &&
                    (*out_dtype)->elsize >= itemsize) {
                return 0;
            }
            descriptor = PyArray_DescrNewFromType(string_type);
            if (descriptor == NULL) {
                goto fail;
            }
            descriptor->elsize = itemsize;
        }
        goto promote_types;
    }

    /* Check if it's a Python scalar */
    descriptor = _array_find_python_scalar_type(obj);
    if (descriptor != NULL) {
        if (string_type) {
            int itemsize;
            PyObject *temp;

            /* dtype is not used in this (string discovery) branch */
            Py_DECREF(descriptor);
            descriptor = NULL;

            if (string_type == NPY_STRING) {
                if ((temp = PyObject_Str(obj)) == NULL) {
                    goto fail;
                }
#if defined(NPY_PY3K)
    #if PY_VERSION_HEX >= 0x03030000
                itemsize = PyUnicode_GetLength(temp);
    #else
                itemsize = PyUnicode_GET_SIZE(temp);
    #endif
#else
                itemsize = PyString_GET_SIZE(temp);
#endif
            }
            else if (string_type == NPY_UNICODE) {
#if defined(NPY_PY3K)
                if ((temp = PyObject_Str(obj)) == NULL) {
#else
                if ((temp = PyObject_Unicode(obj)) == NULL) {
#endif
                    goto fail;
                }
                itemsize = PyUnicode_GET_DATA_SIZE(temp);
#ifndef Py_UNICODE_WIDE
                itemsize <<= 1;
#endif
            }
            else {
                goto fail;
            }
            Py_DECREF(temp);
            if (*out_dtype != NULL &&
                    (*out_dtype)->type_num == string_type &&
                    (*out_dtype)->elsize >= itemsize) {
                return 0;
            }
            descriptor = PyArray_DescrNewFromType(string_type);
            if (descriptor == NULL) {
                goto fail;
            }
            descriptor->elsize = itemsize;
        }
        goto promote_types;
    }

    /* Check if it's an ASCII string */
    if (PyBytes_Check(obj)) {
        int itemsize = PyString_GET_SIZE(obj);

        /* If it's already a big enough string, don't bother type promoting */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_STRING &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        descriptor = PyArray_DescrNewFromType(NPY_STRING);
        if (descriptor == NULL) {
            goto fail;
        }
        descriptor->elsize = itemsize;
        goto promote_types;
    }

    /* Check if it's a Unicode string */
    if (PyUnicode_Check(obj)) {
        int itemsize = PyUnicode_GET_DATA_SIZE(obj);
#ifndef Py_UNICODE_WIDE
        itemsize <<= 1;
#endif

        /*
         * If it's already a big enough unicode object,
         * don't bother type promoting
         */
        if (*out_dtype != NULL &&
                        (*out_dtype)->type_num == NPY_UNICODE &&
                        (*out_dtype)->elsize >= itemsize) {
            return 0;
        }
        descriptor = PyArray_DescrNewFromType(NPY_UNICODE);
        if (descriptor == NULL) {
            goto fail;
        }
        descriptor->elsize = itemsize;
        goto promote_types;
    }

    /* PEP 3118 buffer interface */
    if (PyObject_CheckBuffer(obj) == 1) {
        memset(&buffer_view, 0, sizeof(Py_buffer));
        if (PyObject_GetBuffer(obj, &buffer_view,
                               PyBUF_FORMAT|PyBUF_STRIDES) == 0 ||
            PyObject_GetBuffer(obj, &buffer_view,
                               PyBUF_FORMAT|PyBUF_SIMPLE) == 0) {

            PyErr_Clear();
            descriptor = _descriptor_from_pep3118_format(buffer_view.format);
            PyBuffer_Release(&buffer_view);
            _dealloc_cached_buffer_info(obj);
            if (descriptor) {
                goto promote_types;
            }
        }
        else if (PyObject_GetBuffer(obj, &buffer_view, PyBUF_STRIDES) == 0 ||
                 PyObject_GetBuffer(obj, &buffer_view, PyBUF_SIMPLE) == 0) {

            PyErr_Clear();
            descriptor = PyArray_DescrNewFromType(NPY_VOID);
            descriptor->elsize = buffer_view.itemsize;
            PyBuffer_Release(&buffer_view);
            _dealloc_cached_buffer_info(obj);
            goto promote_types;
        }
        else {
            PyErr_Clear();
        }
    }

    /* The array interface */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array_interface__");
    if (ip != NULL) {
        if (PyDict_Check(ip)) {
            PyObject *typestr;
#if defined(NPY_PY3K)
            PyObject *tmp = NULL;
#endif
            typestr = PyDict_GetItemString(ip, "typestr");
#if defined(NPY_PY3K)
            /* Allow unicode type strings */
            if (typestr && PyUnicode_Check(typestr)) {
                tmp = PyUnicode_AsASCIIString(typestr);
                typestr = tmp;
            }
#endif
            if (typestr && PyBytes_Check(typestr)) {
                descriptor =_array_typedescr_fromstr(PyBytes_AS_STRING(typestr));
#if defined(NPY_PY3K)
                if (tmp == typestr) {
                    Py_DECREF(tmp);
                }
#endif
                Py_DECREF(ip);
                if (descriptor == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }
    else if (PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }


    /* The array struct interface */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array_struct__");
    if (ip != NULL) {
        PyArrayInterface *inter;
        char buf[40];

        if (NpyCapsule_Check(ip)) {
            inter = (PyArrayInterface *)NpyCapsule_AsVoidPtr(ip);
            if (inter->two == 2) {
                PyOS_snprintf(buf, sizeof(buf),
                        "|%c%d", inter->typekind, inter->itemsize);
                descriptor = _array_typedescr_fromstr(buf);
                Py_DECREF(ip);
                if (descriptor == NULL) {
                    goto fail;
                }
                goto promote_types;
            }
        }
        Py_DECREF(ip);
    }
    else if (PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    /* The old buffer interface */
#if !defined(NPY_PY3K)
    if (PyBuffer_Check(obj)) {
        dtype = PyArray_DescrNewFromType(NPY_VOID);
        if (dtype == NULL) {
            goto fail;
        }
        dtype->elsize = Py_TYPE(obj)->tp_as_sequence->sq_length(obj);
        PyErr_Clear();
        goto promote_types;
    }
#endif

    /* The __array__ attribute */
    ip = PyArray_LookupSpecial_OnInstance(obj, "__array__");
    if (ip != NULL) {
        Py_DECREF(ip);
        ip = PyObject_CallMethod(obj, "__array__", NULL);
        if(ip && PyArray_Check(ip)) {
            descriptor = PyArray_DESCR((PyArrayObject *)ip);
            Py_INCREF(descriptor);
            Py_DECREF(ip);
            goto promote_types;
        }
        Py_XDECREF(ip);
        if (PyErr_Occurred()) {
            goto fail;
        }
    }
    else if (PyErr_Occurred()) {
        PyErr_Clear(); /* TODO[gh-14801]: propagate crashes during attribute access? */
    }

    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, and obj isn't a sequence-like object, the output
     * dtype should be either OBJECT or a user-defined type.
     *
     * Note that some libraries define sequence-like classes but want them to
     * be treated as objects, and they expect numpy to treat it as an object if
     * __len__ is not defined.
     */
    if (maxdims == 0 || !PySequence_Check(obj) || PySequence_Size(obj) < 0) {
        /* clear any PySequence_Size error which corrupts further calls */
        PyErr_Clear();

        if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
            Py_XDECREF(*out_dtype);
            *out_dtype = PyArray_DescrFromType(NPY_OBJECT);
            if (*out_dtype == NULL) {
                return -1;
            }
        }
        return 0;
    }

    /*
     * The C-API recommends calling PySequence_Fast before any of the other
     * PySequence_Fast* functions. This is required for PyPy
     */
    seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        goto fail;
    }

    /* Recursive case, first check the sequence contains only one type */
    size = PySequence_Fast_GET_SIZE(seq);
    /* objects is borrowed, do not release seq */
    objects = PySequence_Fast_ITEMS(seq);
    common_type = size > 0 ? Py_TYPE(objects[0]) : NULL;
    for (i = 1; i < size; ++i) {
        if (Py_TYPE(objects[i]) != common_type) {
            common_type = NULL;
            break;
        }
    }

    /* all types are the same and scalar, one recursive call is enough */
    if (common_type != NULL && !string_type &&
            (common_type == &PyFloat_Type ||
/* TODO: we could add longs if we add a range check */
#if !defined(NPY_PY3K)
             common_type == &PyInt_Type ||
#endif
             common_type == &PyBool_Type ||
             common_type == &PyComplex_Type)) {
        size = 1;
    }

    /* Recursive call for each sequence item */
    for (i = 0; i < size; ++i) {
        int res = PyArray_DTypeFromObjectHelper(objects[i], maxdims - 1,
                                                out_dtype, string_type);
        if (res < 0) {
            Py_DECREF(seq);
            goto fail;
        }
        else if (res > 0) {
            Py_DECREF(seq);
            return res;
        }
    }

    Py_DECREF(seq);

    return 0;


promote_types:
    /* Set 'out_dtype' if it's NULL */
    if (*out_dtype == NULL) {
        if (!string_type && descriptor->type_num == NPY_STRING) {
            Py_DECREF(descriptor);
            return RETRY_WITH_STRING;
        }
        if (!string_type && descriptor->type_num == NPY_UNICODE) {
            Py_DECREF(descriptor);
            return RETRY_WITH_UNICODE;
        }
        *out_dtype = descriptor;
        return 0;
    }
    /* Do type promotion with 'out_dtype' */
    else {
        PyArray_Descr *res_dtype = PyArray_PromoteTypes(descriptor, *out_dtype);
        Py_DECREF(descriptor);
        if (res_dtype == NULL) {
            goto fail;
        }
        if (!string_type &&
                res_dtype->type_num == NPY_UNICODE &&
                (*out_dtype)->type_num != NPY_UNICODE) {
            Py_DECREF(res_dtype);
            return RETRY_WITH_UNICODE;
        }
        if (!string_type &&
                res_dtype->type_num == NPY_STRING &&
                (*out_dtype)->type_num != NPY_STRING) {
            Py_DECREF(res_dtype);
            return RETRY_WITH_STRING;
        }
        Py_DECREF(*out_dtype);
        *out_dtype = res_dtype;
        return 0;
    }

fail:
    Py_XDECREF(*out_dtype);
    *out_dtype = NULL;
    return -1;
}

#undef RETRY_WITH_STRING
#undef RETRY_WITH_UNICODE

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


// TODO: We want two different modes here to catch the special cases where
//       used to use make use of value based casting (i.e. 0D arrays), and
//       the other one for array coercion, where we do not do this normally.
//       (although possibly that part will naturally go somewhere else.)
/*
 * Internal helper to find the correct DType (class). Must be called with
 * `curr_dim = 0`. Returns the maximum reached depth and negative number
 * on failure. `out_dtype` is NULL on error, otherwise a reference to a DType
 * class. Note that the maximum reached depth is not changed if empty sequences
 * are found.
 * The caller may get an abstract dtype returned, at which point it is may
 * be necessary to convert it.
 */
NPY_NO_EXPORT int
PyArray_DiscoverDTypeFromObject(
        PyObject *obj, int max_dims, int curr_dims,
        PyArray_DTypeMeta **out_dtype)
{
    PyArray_DTypeMeta *dtype = NULL;
    PyArray_Descr *descriptor = NULL;

    static PyTypeObject *prev_type;
    static PyArray_DTypeMeta *prev_dtype;
    if (curr_dims == 0) {
        /* Clear the static cache of which types we have already seen */
        prev_type = NULL;
        prev_dtype = NULL;
    }

    if (prev_type == Py_TYPE(obj)) {
        dtype = prev_dtype;
        Py_INCREF(dtype);
    }
    else {
        dtype = (PyArray_DTypeMeta *) PyDict_GetItem(
                PyArrayDTypeMeta_associated_types, (PyObject *) Py_TYPE(obj));
    }
    if (dtype != NULL) {
        /* Cache that we have discovered the dtype class: */
        prev_type = Py_TYPE(obj);
        prev_dtype = dtype;
        // TODO: At least if the below does not return an abstract dtype
        //       we could cache that directly. Otherwise, it is possible,
        //       but requires that the returned dtype implements discovery.
        if (dtype->abstract) {
            if (dtype->dt_slots->requires_pyobject_for_discovery) {
                dtype = dtype->dt_slots->discover_dtype_from_pytype(dtype, obj);
            }
            else {
                dtype = dtype->dt_slots->discover_dtype_from_pytype(
                        dtype, (PyObject *)Py_TYPE(obj));
            }
        }
        goto promote_types;
    }

    /* Check if it's an ndarray */
    if (PyArray_Check(obj)) {
        descriptor = PyArray_DESCR((PyArrayObject *)obj);
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        curr_dims += PyArray_NDIM(obj);
        goto promote_types;
    }

    /* Check if it's a NumPy scalar */
    // TODO: Should be found before, unless a subclass, wihich is no good?
    if (PyArray_IsScalar(obj, Generic)) {
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
    // TODO: Deprecate!
    /* Check if it's a Python scalar */
    descriptor = _array_find_python_scalar_type(obj);
    if (descriptor != NULL) {
        dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
        Py_INCREF(dtype);
        Py_DECREF(descriptor);
        prev_type = Py_TYPE(obj);
        prev_dtype = dtype;
        goto promote_types;
    }

    /* Check if it's an ASCII string */
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

    {
        PyObject *tmp = _array_from_array_like(obj,  /* requested_dtype */ NULL,
                                               NPY_FALSE, /* context */ NULL);
        if (tmp != Py_NotImplemented) {
            descriptor = PyArray_DESCR(tmp);
            curr_dims += PyArray_NDIM(tmp);
            Py_DECREF(tmp);
            dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
            Py_INCREF(dtype);
            goto promote_types;
        }
        Py_DECREF(tmp);
    }
    /*
     * If we reached the maximum recursion depth without hitting one
     * of the above cases, and obj isn't a sequence-like object, the output
     * dtype should be either OBJECT or a user-defined type.
     *
     * Note that some libraries define sequence-like classes but want them to
     * be treated as objects, and they expect numpy to treat it as an object if
     * __len__ is not defined.
     */
    if (max_dims - curr_dims == 0 ||
                !PySequence_Check(obj) || PySequence_Size(obj) < 0) {
        /* clear any PySequence_Size error which corrupts further calls */
        PyErr_Clear();

        if (*out_dtype == NULL || (*out_dtype)->type_num != NPY_OBJECT) {
            // TODO: We may want to add a deprecation warning in this path,
            //       in generally, object should probably never happen without
            //       the user asking for it specifically.
            Py_XDECREF(*out_dtype);
            descriptor = PyArray_DescrFromType(NPY_OBJECT);
            dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
            Py_INCREF(dtype);
            Py_DECREF(descriptor);
            if (*out_dtype == NULL) {
                return -1;
            }
        }
        return curr_dims;
    }

    /*
     * The C-API recommends calling PySequence_Fast before any of the other
     * PySequence_Fast* functions. This is required for PyPy
     */
    PyObject *seq = PySequence_Fast(obj, "Could not convert object to sequence");
    if (seq == NULL) {
        goto fail;
    }

    // TODO: Old case checked if there was only a single type present
    //       to optimize (super fast path for floats, bools, and complex)
    Py_ssize_t  size = PySequence_Fast_GET_SIZE(seq);
    PyObject **objects = PySequence_Fast_ITEMS(seq);
    /* Recursive call for each sequence item */
    if (size == 0) {
        // TODO,XXX: This blocks `np.array([np.empty((0,3)), []])`, but that
        //           is legacy behaviour and probably OK...
        max_dims = curr_dims + 1;
    }
    npy_bool max_dims_found = NPY_FALSE;
    for (Py_ssize_t i = 0; i < size; ++i) {
        /*
         * Discover each item, which could also be a sequence, so we store
         * the smallest dimension that was found (in principle allowing
         * ragged arrays). Do this by adjust max_dims.
         * TODO: Check that this is really identical (enough)
         */
        int new_dims = PyArray_DiscoverDTypeFromObject(
                objects[i], max_dims, curr_dims + 1, out_dtype);
        if (new_dims != max_dims) {
            max_dims = new_dims < max_dims ? new_dims : max_dims;
            if (max_dims_found) {
                /* This is a ragged array */
                // TODO: Set deprecation warning or maybe error message?!
                Py_XDECREF(*out_dtype);
                descriptor = PyArray_DescrFromType(NPY_OBJECT);
                dtype = (PyArray_DTypeMeta *)Py_TYPE(descriptor);
                Py_INCREF(dtype);
                Py_DECREF(descriptor);
            }
            max_dims_found = NPY_TRUE;
        }
        if (new_dims < 0) {
            Py_DECREF(seq);
            goto fail;
        }
    }
    Py_DECREF(seq);
    return max_dims;

promote_types:
    if (*out_dtype == NULL) {
        *out_dtype = dtype;
        return curr_dims;
    }
    Py_SETREF(*out_dtype, PyArray_PromoteDTypes(*out_dtype, dtype));
    Py_DECREF(dtype);
    if (*out_dtype == NULL) {
        return -1;
    }
    // TODO: Only the single object/0D case may use minimal dtype, so in
    //       principle here (and even above), we could convert abstract
    //       DTypes using their default (or even tell them to give the default).
    return curr_dims;

fail:
    Py_XDECREF(*out_dtype);
    *out_dtype = NULL;
    return -1;
}


