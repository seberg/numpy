#include "ufunc_impl.h"
#include "extobj.h"


static int
ufuncimpl_setup_clear_fp(
        PyUFuncImplObject *ufunc_impl, PyUFuncObject * ufunc,
        void **data, PyObject *extobj, int errormask) {
    npy_clear_floatstatus_barrier((char*)&ufunc_impl);
    return 0;
}


static int
ufuncimpl_teardown_check_floatstatus(
        PyUFuncImplObject *ufunc_impl, PyUFuncObject * ufunc,
        void **data, PyObject *extobj, int errormask) {
    return _check_ufunc_fperr(errormask, extobj, ufunc->name);
}


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
    new_impl->adapt_dtype_func = ufuncimpl_adapt_dtype_from_pyfunc;
    new_impl->adapt_dtype_pyfunc = callable;
    return (PyObject *)new_impl;
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
