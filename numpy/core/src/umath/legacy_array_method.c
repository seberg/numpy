/*
 * This file defines most of the machinery in order to wrap legacy style
 * ufunc loops into new style arraymethods.
 */

#include <numpy/ndarraytypes.h>
#include <array_method.h>
#include <dtype_transfer.h>
#include "umathmodule.h"
#include "legacy_array_method.h"


typedef struct {
    NpyAuxData base;
    /* The legacy loop and additional user data: */
    PyUFuncGenericFunction loop;
    void *user_data;
    /* Whether to check for PyErr_Occurred(), must require GIL if used */
    int pyerr_check;
} legacy_array_method_auxdata;


/* Use a free list, since we should normally only need one at a time */
#define NPY_LOOP_DATA_CACHE_SIZE 5
static int loop_data_num_cached = 0;
static  legacy_array_method_auxdata *loop_data_cache[NPY_LOOP_DATA_CACHE_SIZE];


static void
legacy_array_method_auxdata_free(NpyAuxData *data)
{
    if (loop_data_num_cached < NPY_LOOP_DATA_CACHE_SIZE) {
        loop_data_cache[loop_data_num_cached] = (
                (legacy_array_method_auxdata *)data);
        loop_data_num_cached++;
    }
    else {
        PyMem_Free(data);
    }
}

#undef NPY_LOOP_DATA_CACHE_SIZE


NpyAuxData *
get_new_loop_data(
        PyUFuncGenericFunction loop, void *user_data, int pyerr_check)
{
    legacy_array_method_auxdata *data;
    if (NPY_LIKELY(loop_data_num_cached > 0)) {
        loop_data_num_cached--;
        data = loop_data_cache[loop_data_num_cached];
    }
    else {
        data = PyMem_Malloc(sizeof(legacy_array_method_auxdata));
        if (data == NULL) {
            return NULL;
        }
        data->base.free = legacy_array_method_auxdata_free;
        data->base.clone = NULL;  /* no need for cloning (at least for now) */
    }
    data->loop = loop;
    data->user_data = user_data;
    data->pyerr_check = pyerr_check;
    return (NpyAuxData *)data;
}


/*
 * This is a thin wrapper around the legacy loop signature.
 */
static int
generic_wrapped_legacy_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *auxdata)
{
    legacy_array_method_auxdata *ldata = (legacy_array_method_auxdata *)auxdata;

    ldata->loop((char **)data, dimensions, strides, ldata->user_data);
    if (ldata->pyerr_check && PyErr_Occurred()) {
        return -1;
    }
    return 0;
}


/*
 * It just seems to annoying to make work correctly, there are two main
 * problems: First, the resolver really would like the ufunc to be passed in
 * (which we could store). Second, the resolver is currently passed the
 * full array objects (it is also passed the casting safety instead of
 * returning it).
 * Overall, the discrepancy is just a bit much, so if this is actually called
 * we are out of luck, and otherwise we use it to signal the legacy fallback
 * path.
 */
NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]),
        PyArray_Descr *NPY_UNUSED(given_descrs[]),
        PyArray_Descr *NPY_UNUSED(loop_descrs[]))
{
    PyErr_SetString(PyExc_RuntimeError,
            "cannot use legacy wrapping ArrayMethod without calling the ufunc "
            "itself.  If this error is hit, the solution will be to port the "
            "legacy ufunc loop implementation to the new API.");
    return -1;
}


/*
 * This function grabs the legacy inner-loop.  If this turns out to be slow
 * we could probably cache it (with some care).
 */
static int
get_wrapped_legacy_ufunc_loop(PyArrayMethod_Context *context,
        int aligned, int move_references,
        npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    assert(aligned);
    assert(!move_references);

    if (NPY_UNLIKELY(context->caller == NULL ||
            PyObject_TypeCheck(context->caller, &PyUFunc_Type))) {
        PyErr_Format(PyExc_RuntimeError,
                "cannot call %s without its ufunc as caller context.",
                context->method->name);
        return -1;
    }
    PyUFuncObject *ufunc = (PyUFuncObject *)context->caller;
    void *user_data;
    int needs_api = 0;

    PyUFuncGenericFunction loop = NULL;
    /* Note that `needs_api` is not reliable (it was in fact unused normally) */
    if (ufunc->legacy_inner_loop_selector(ufunc,
            context->descriptors, &loop, &user_data, &needs_api) < 0) {
        return -1;
    }
    *flags = context->method->flags & NPY_METH_RUNTIME_FLAGS;
    if (needs_api) {
        *flags |= NPY_METH_REQUIRES_PYAPI;
    }

    *out_loop = &generic_wrapped_legacy_loop;
    *out_transferdata = get_new_loop_data(
            loop, user_data, (*flags & NPY_METH_REQUIRES_PYAPI) != 0);
    return 0;
}


/*
 * Get the unbound ArrayMethod which wraps the instances of the ufunc.
 * Note that this function stores the result on the ufunc and then only
 * returns the same one.
 */
NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[])
{
    if (ufunc->_legacy_array_method != NULL) {
        Py_INCREF(ufunc->_legacy_array_method);
        return (PyArrayMethodObject *)ufunc->_legacy_array_method;
    }

    char method_name[101];
    const char *name = ufunc->name ? ufunc->name : "<unknown>";
    snprintf(method_name, 100, "legacy_ufunc_wrapper_for_%s", name);

    /*
     * Assume that we require the Python API when any of the (legacy) dtypes
     * flags it.
     */
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    for (int i = 0; i < ufunc->nin+ufunc->nout; i++) {
        if (signature[i]->singleton->flags & NPY_NEEDS_PYAPI) {
            flags &= NPY_METH_REQUIRES_PYAPI;
        }
    }

    PyType_Slot slots[] = {
        {NPY_METH_resolve_descriptors, &wrapped_legacy_resolve_descriptors},
        {NPY_METH_get_loop, &get_wrapped_legacy_ufunc_loop},
        {0, NULL},
    };

    PyArrayMethod_Spec spec = {
        .name = method_name,
        .nin = ufunc->nin,
        .nout = ufunc->nout,
        .dtypes = signature,
        .flags = flags,
        .slots = slots,
    };

    PyBoundArrayMethodObject *bound_res = PyArrayMethod_FromSpec_int(&spec, 1);
    if (bound_res == NULL) {
        return NULL;
    }
    PyArrayMethodObject *res = bound_res->method;
    Py_INCREF(res);
    Py_DECREF(bound_res);
    return res;
}