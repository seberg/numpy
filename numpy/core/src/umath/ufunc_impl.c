#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#include "npy_config.h"
#include "alloc.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "extobj.h"

#include "ufunc_impl.h"


static int
ufuncimpl_setup_clear_fp(
        PyUFuncImplObject *ufunc_impl, PyUFuncObject * ufunc,
        void **data, PyObject *extobj, int errormask) {
    npy_clear_floatstatus_barrier((char*)&ufunc_impl);
    return 0;
}


/*
static int
ufuncimpl_teardown_check_floatstatus(
        PyUFuncImplObject *ufunc_impl, PyUFuncObject * ufunc,
        void **data, PyObject *extobj, int errormask) {
    return _check_ufunc_fperr(errormask, extobj, ufunc->name);
}
*/


static int
ufuncimpl_teardown_check_pyexc_floatstatus(
        PyUFuncImplObject *ufunc_impl, PyUFuncObject * ufunc,
        void **data, PyObject *extobj, int errormask) {
    if (PyErr_Occurred() ||
        (_check_ufunc_fperr(errormask, extobj, ufunc->name) < 0)) {
        return -1;
    }
    return 0;
}

/******************************************************************************/


static int
default_ufunc_adapt_function(PyUFuncImplObject *self,
        PyArray_Descr **descr, PyArray_Descr **out_descr,
        NPY_CASTING casting)
{
    /*
     * For typical ufuncs all descriptors being used must be native byte order.
     * There are two interesting possibilities:
     *   1. We try to preserve metadata on the dtypes. The old style type
     *      resolution function does this only for the first input.
     *   2. We need to call the old style type resolution for flexible dtypes.
     * Before returning, we also need to validate the casting.
     */
    int i = 0;
    int nop = self->nin + self->nout;

    for (i = 0; i < self->nin; i++) {
        PyArray_DTypeMeta *given_dt = self->dtype_signature[i];
        PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)Py_TYPE(descr[i]);

        if (given_dt != dt) {
            /* Casting is necessary, so use the default descriptor. */
            out_descr[i] = given_dt->dt_slots->default_descr(given_dt);
        }
        else if (PyDataType_ISNOTSWAPPED(descr[i])) {
            /* No casting necessary, fast path reusing input. */
            out_descr[i] = descr[i];
            Py_INCREF(out_descr[i]);
        }
        else {
            /* Same as above, but ensure input is native */
            out_descr[i] = dt->dt_slots->ensure_native(descr[i]);
            if (out_descr[i] == NULL) {
                goto fail;
            }
        }
    }
    for (i = self->nin; i < nop; i++) {
        PyArray_DTypeMeta *given_dt = self->dtype_signature[i];
        if (descr[i] == NULL) {
            /* The first branch here preserves metadata of first input */
            if (given_dt == ((PyArray_DTypeMeta *) Py_TYPE(out_descr[0]))) {
                out_descr[i] = out_descr[0];
            } else {
                out_descr[i] = given_dt->dt_slots->default_descr(given_dt);
            }
            if (out_descr[i] == NULL) {
                goto fail;
            }
        }
        else {
            PyArray_DTypeMeta *dt = (PyArray_DTypeMeta *)Py_TYPE(descr[i]);

            if (given_dt != dt) {
                /* Casting is necessary, so use the default descriptor. */
                out_descr[i] = given_dt->dt_slots->default_descr(given_dt);
            }
            else if (PyDataType_ISNOTSWAPPED(descr[i])) {
                /* No casting necessary, fast path reusing input. */
                out_descr[i] = descr[i];
                Py_INCREF(out_descr[i]);
            }
            else {
                /* Same as above, but ensure input is native */
                out_descr[i] = dt->dt_slots->ensure_native(descr[i]);
                if (out_descr[i] == NULL) {
                    goto fail;
                }
            }
        }
    }

    // TODO: previously we had the ufunc here if a single loop
    //       could attach to multiple ufuncs this info may not be
    //       available here. Alternatively we could pass in the
    //       ufunc, but say that it may be NULL.
    for (i = 0; i < nop; i++) {
           if (i < self->nin) {
               if (!PyArray_CanCastTypeTo(descr[i], out_descr[i], casting)) {
                   PyErr_SetString(PyExc_TypeError,
                           "Cannot cast input (error message needs improvement)");
                   i = nop;
                   goto fail;
               }
           }
           else if (descr[i] != NULL) {
               /* If the input is NULL, we can assume casting is fine */
               if (!PyArray_CanCastTypeTo(out_descr[i], descr[i], casting)) {
                   PyErr_SetString(PyExc_TypeError,
                                   "Cannot cast output (error message needs improvement)");
                   i = nop;
                   goto fail;
               }
           }
    }
    return 0;

fail:
    for (int j; j < i; j++) {
        Py_DECREF(out_descr[j]);
    }
    return -1;
}



NPY_NO_EXPORT PyObject *
ufuncimpl_legacy_new(PyUFuncObject *ufunc, PyArray_DTypeMeta **dtypes)
{
    PyUFuncImplObject *ufunc_impl = (PyUFuncImplObject *)PyObject_New(
            PyUFuncImplObject, &PyUFuncImpl_Type);
    if (ufunc_impl == NULL) {
        return NULL;
    }
    ufunc_impl->dtype_signature = malloc(ufunc->nargs * sizeof(PyObject *));
    if (ufunc_impl->dtype_signature == NULL) {
        PyObject_FREE(ufunc_impl);
        return NULL;
    }
    memset(ufunc_impl->dtype_signature, 0, ufunc->nargs * sizeof(PyObject *));

    ufunc_impl->is_legacy_wrapper = NPY_TRUE;
    ufunc_impl->nin = ufunc->nin;
    ufunc_impl->nout = ufunc->nout;

    ufunc_impl->identity = ufunc->identity;
    Py_XINCREF(ufunc->identity_value);
    ufunc_impl->identity_value = ufunc->identity_value;
    if (ufunc->op_flags != NULL) {
        ufunc_impl->op_flags = PyArray_malloc(sizeof(npy_uint32)*ufunc->nargs);
        memcpy(ufunc_impl->op_flags, ufunc->op_flags,
               sizeof(npy_uint32)*ufunc->nargs);
    }
    else {
        ufunc_impl->op_flags = NULL;
    }
    ufunc_impl->iter_flags = ufunc->iter_flags;

    // TODO: This function does not work for flexible dtypes
    //       (specifically, it does not work for datetimes, where we would
    //       have to use the olds tyle TypeResolution function).
    ufunc_impl->adapt_dtype_func = default_ufunc_adapt_function;
    ufunc_impl->adapt_dtype_pyfunc = NULL;

    /*
     * The correct loop will be fetched on every execution (or rather
     * every time the resolution finished). This is to ensure that we pick
     * up changes that use the loop replacement API. It should also make
     * things more robust towards downstream modifying the ufunc object
     * directly.
     */
    ufunc_impl->innerloop = NULL;
    ufunc_impl->innerloopdata = NULL;
    ufunc_impl->needs_api = 1;

    for (int i = 0; i < ufunc->nargs; i++) {
        ufunc_impl->dtype_signature[i] = dtypes[i];
        Py_INCREF(ufunc_impl->dtype_signature[i]);
    }

    /* TODO: What the heck to do about errors during casting? */
    /*
     * Python errors could occur, check everything. In some cases
     * we may be able to optimize this, especially to not check floating
     * point errors.
     */
    ufunc_impl->setup = ufuncimpl_setup_clear_fp;
    ufunc_impl->teardown = ufuncimpl_teardown_check_pyexc_floatstatus;

    return (PyObject *)ufunc_impl;
}



static void
ufuncimpl_dealloc(PyUFuncImplObject *ufunc_impl)
{
    /* TODO: May need cyclic GC support? */

    if (ufunc_impl->identity == PyUFunc_IdentityValue) {
        Py_DECREF(ufunc_impl->identity_value);
    }
    Py_XDECREF(ufunc_impl->adapt_dtype_pyfunc);
    PyDataMem_FREE(ufunc_impl->dtype_signature);
}


/*
static int
ufuncimpl_adapt_dtype_from_pyfunc(PyUFuncImplObject *ufunc_impl,
                                  PyArray_Descr **dtypes) {
    PyObject *dtypes_tuple;
    PyObject *new_tuple;
    int nop = ufunc_impl->nin + ufunc_impl->nout;


    dtypes_tuple = PyTuple_New(nop);
    // TODO: Error Check.
    for (Py_ssize_t i = 0; i < nop; i++) {
        PyObject *tmp = (dtypes[i] != NULL) ? (PyObject *)dtypes[i] : Py_None;
        Py_INCREF(tmp);
        PyTuple_SET_ITEM(dtypes_tuple, i, tmp);
    }

    new_tuple = PyObject_CallFunctionObjArgs(
            ufunc_impl->adapt_dtype_pyfunc,
            ufunc_impl, dtypes_tuple, NULL);
    Py_DECREF(dtypes_tuple);
    if (new_tuple == NULL) {
        return -1;
    }

    if (!PyTuple_CheckExact(new_tuple)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "ufunc dtype adaption returned must return a tuple.");
        return -1;
    }

    if (PyTuple_Size(new_tuple) != nop) {
        PyErr_SetString(PyExc_RuntimeError,
                        "ufunc dtype adaption returned wrong number of arguments.");
        return -1;
    }

    for (int i = 0; i < nop; i++) {
        if (!PyArray_DescrCheck(PyTuple_GET_ITEM(new_tuple, i))) {
            PyErr_SetString(PyExc_RuntimeError,
                            "ufunc dtype adaption returned returned non-dtype.");
            return -1;
        }
        Py_SETREF(dtypes[i],
                  (PyArray_Descr *)PyTuple_GET_ITEM(new_tuple, i));
        Py_INCREF(dtypes[i]);
    }
    return 0;
}
*/


static PyUFuncImplObject *
ufuncimpl_copy(PyUFuncImplObject *ufunc_impl) {
    PyUFuncImplObject *new_ufunc_impl;
    int nops = ufunc_impl->nin + ufunc_impl->nout;

    new_ufunc_impl = PyObject_New(PyUFuncImplObject, &PyUFuncImpl_Type);

    new_ufunc_impl->nin = ufunc_impl->nin;
    new_ufunc_impl->nout = ufunc_impl->nout;

    new_ufunc_impl->identity = ufunc_impl->identity;
    Py_XINCREF(ufunc_impl->identity_value);
    new_ufunc_impl->identity_value = ufunc_impl->identity_value;
    if (ufunc_impl->op_flags != NULL) {
        // TODO: Can error:
        new_ufunc_impl->op_flags = PyArray_malloc(sizeof(npy_uint32) * nops);
        memcpy(new_ufunc_impl->op_flags, ufunc_impl->op_flags,
               sizeof(npy_uint32) * nops);
    }
    else {
        new_ufunc_impl->op_flags = NULL;
    }
    new_ufunc_impl->iter_flags = ufunc_impl->iter_flags;

    new_ufunc_impl->innerloop = ufunc_impl->innerloop;
    new_ufunc_impl->innerloopdata = ufunc_impl->innerloopdata;
    new_ufunc_impl->needs_api = ufunc_impl->needs_api;

    /* Default is not to have one (we do not currently) */
    new_ufunc_impl->adapt_dtype_func = ufunc_impl->adapt_dtype_func;
    new_ufunc_impl->adapt_dtype_pyfunc = ufunc_impl->adapt_dtype_pyfunc;

    new_ufunc_impl->setup = ufunc_impl->setup;
    new_ufunc_impl->teardown = ufunc_impl->teardown;
    Py_XINCREF(new_ufunc_impl->adapt_dtype_pyfunc);

    return new_ufunc_impl;
}


static PyObject *
ufuncimpl_replaced_dtype_adapt(PyUFuncImplObject *self, PyObject *callable)
{
    PyUFuncImplObject *new_impl = ufuncimpl_copy(self);

    Py_INCREF(callable);
    //new_impl->adapt_dtype_func = ufuncimpl_adapt_dtype_from_pyfunc;
    new_impl->adapt_dtype_pyfunc = callable;
    return (PyObject *)NULL;
}


static struct PyMethodDef ufuncimpl_methods[] = {
        {"replaced_dtype_adapt",
                (PyCFunction)ufuncimpl_replaced_dtype_adapt,
                     METH_O, NULL },
        {NULL, NULL, 0, NULL}  /* sentinel */
};

/******************************************************************************
 ***                        UFUNC IMPL Type OBJECT                          ***
 *****************************************************************************/

NPY_NO_EXPORT PyTypeObject PyUFuncImpl_Type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name ="numpy._ufuncimpl",
        .tp_basicsize = sizeof(PyUFuncImplObject),
        .tp_dealloc = (destructor)ufuncimpl_dealloc,
        .tp_methods = ufuncimpl_methods,
};
