#ifndef _NPY_UFUNC_IMPL_H
#define _NPY_UFUNC_IMPL_H


/* Forward declaration for the dtype adaption function typedef */
struct _tagPyUFuncImplObject;


NPY_NO_EXPORT PyObject *
ufuncimpl_legacy_new(PyUFuncObject *ufunc, PyArray_DTypeMeta **dtypes);


// TODO: Synchronize with CastingImpl adaptation function, which should
//       just be this one! (`adjust_descriptors_func`)
typedef int (PyUfuncAdaptFlexibleDTypes) (
        struct _tagPyUFuncImplObject *ufunc_impl,
        PyArray_Descr **dtypes, PyArray_Descr **out_descrs,
        NPY_CASTING casting);

typedef int (PyUfuncImplSetupFunc) (
        // TODO: Should get a whole lot more stuff (probably just about everything)
        //       Possibly, it would be one function even handling everything.
        struct _tagPyUFuncImplObject *ufunc_impl, PyUFuncObject *ufunc,
        void **data, PyObject *extobj, int errormask);

typedef int (PyUfuncImplTeardownFunc) (
        struct _tagPyUFuncImplObject *ufunc_impl, PyUFuncObject *ufunc,
        void **data, PyObject *extobj, int errormask);

typedef struct _tagPyUFuncImplObject {
    PyObject_HEAD
    /*
     * Identity for reduction, any of PyUFunc_One, PyUFunc_Zero
     * PyUFunc_MinusOne, PyUFunc_None, PyUFunc_ReorderableNone,
     * PyUFunc_IdentityValue.
     */
    int identity;

    int nin;
    int nout;

    int needs_api;
    /*
     * TODO: Probably needs most/all ufunc slots to assert compatibility
     *       for the python side API.
     */

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

    PyUFuncGenericFunction innerloop;
    void *innerloopdata;

    /*
     * For flexible dtypes, the output data type cannot be cached,
     * so it needs to be set after caching is done.
     */
    PyUfuncAdaptFlexibleDTypes *adapt_dtype_func;
    PyObject *adapt_dtype_pyfunc;

    PyUFuncObject *bound_ufunc;

    PyUfuncImplSetupFunc *setup;
    PyUfuncImplTeardownFunc *teardown;
    PyArray_DTypeMeta **dtype_signature;

    npy_bool is_legacy_wrapper;
} PyUFuncImplObject;


#endif /*_NPY_UFUNC_IMPL_H */
