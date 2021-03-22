/*
 * This header exports the new experimental DType API as proposed in
 * NEPs 41 to 43.  For background, please check these NEPs.  Otherwise,
 * this header also serves as documentation at the time being.
 *
 * Please do not hesitate to contact @seberg with questions.  This is
 * developed together with https://github.com/numpy/experimental_user_dtypes
 * and those interested in experimenting are encouraged to contribute there.
 *
 * To use the functions defined in the header call::
 *
 *     if (import_experimental_dtype_api(version) < 0) {
 *         return NULL;
 *     }
 *
 * in your module init.  (A version mismatch will be reported, just update
 * to the correct one, this will alert you of possible changes.)
 *
 * WARNING
 * =======
 *
 * By using this header, you understand that this is a fully experimental
 * exposure.  Details are expected to change, and some options may have no
 * effect.  (Please contact @seberg if you have questions!)
 * Further, a DType created using this API/header should still be expected
 * to be incompatible with some functionality inside and outside of NumPy.
 * In this case crashes must be expected.  Please report any such problems
 * so that they can be fixed before final exposure.
 * Furthermore, expect missing checks for programming errors which the final
 * API is expected to have.
 *
 * Symbols with a leading underscore are likely to not be included in the
 * first public version, if these are central to your use-case, please let
 * us know, so that we can reconsidered.
 *
 * "Array-like" consumer API not yet under considerations
 * ======================================================
 *
 * The new DType API is designed in a way to make it potentially useful for
 * alternative "array-like" implementations.  This will require careful
 * exposure of details and functions and is not part of this experimental API.
 */

#ifndef _NPY_EXPERIMENTAL_DTYPE_API_H
#define _NPY_EXPERIMENTAL_DTYPE_API_H

#include <Python.h>
#include "ndarraytypes.h"


/*
 * Just a hack so I don't forget importing as much myself, I spend way too
 * much time noticing it.  (maybe we should do this for NumPy...)
 */
static void
__not_imported(void)
{
    printf("*****\nCritical error, dtype API not imported\n*****\n");
}
static void *__uninitialized_table[] = {
        &__not_imported, &__not_imported, &__not_imported, &__not_imported};


static void **__experimental_dtype_api_table = __uninitialized_table;

/*
 * ******************************************************
 *                  ArrayMethod API
 * ******************************************************
 */
typedef enum {
    /* Flag for whether the GIL is required */
            NPY_METH_REQUIRES_PYAPI = 1 << 1,
    /*
     * Some functions cannot set floating point error flags, this flag
     * gives us the option (not requirement) to skip floating point error
     * setup/check. No function should set error flags and ignore them
     * since it would interfere with chaining operations (e.g. casting).
     */
            NPY_METH_NO_FLOATINGPOINT_ERRORS = 1 << 2,
    /* Whether the method supports unaligned access (not runtime) */
            NPY_METH_SUPPORTS_UNALIGNED = 1 << 3,

    /* All flags which can change at runtime */
            NPY_METH_RUNTIME_FLAGS = (
            NPY_METH_REQUIRES_PYAPI |
            NPY_METH_NO_FLOATINGPOINT_ERRORS),
} NPY_ARRAYMETHOD_FLAGS;


/*
 * The main object for creating a new ArrayMethod. We use the typical `slots`
 * mechanism used by the Python limited API (see below for the slot defs).
 */
typedef struct {
    const char *name;
    int nin, nout;
    NPY_CASTING casting;
    NPY_ARRAYMETHOD_FLAGS flags;
    PyObject **dtypes;  /* array of DType class objects */
    PyType_Slot *slots;
} PyArrayMethod_Spec;


typedef PyObject *_arraymethod_fromspec_func(PyArrayMethod_Spec *spec);
#define PyArrayMethod_FromSpec \
    (*(_arraymethod_fromspec_func *)(__experimental_dtype_api_table[0]))


/*
 * Additionally to the normal casting levels, NPY_CAST_IS_VIEW indicates
 * that no cast operation is necessary at all (although a copy usually will be)
 */
#define NPY_CAST_IS_VIEW _NPY_CAST_IS_VIEW

/*
 * The resolve descriptors function, must be able to handle NULL values for
 * all output (but not input) `given_descrs` and fill `loop_descrs`.
 * Return -1 on error or 0 if the operation is not possible without an error
 * set.  (This may still be in flux.)
 * Otherwise must return the "casting safety", for normal functions, this is
 * almost always "safe" (or even "equivalent"?).
 *
 * `resolve_descriptors` is optional if all output DTypes are non-parametric.
 */
#define NPY_METH_resolve_descriptors 1
typedef NPY_CASTING (resolve_descriptors_function)(
        /* "method" is currently opaque (necessary e.g. to wrap Python) */
        PyObject *method,
        /* DTypes the method was created for */
        PyObject **dtypes,
        /* Input descriptors (instances).  Outputs may be NULL. */
        PyArray_Descr **given_descrs,
        /* Exact loop descriptors to use, must not hold references on error */
        PyArray_Descr **loop_descrs);

/* NOT public. Signature is expected to change and not included here */
#define _NPY_METH_get_loop 2

/*
 * Current public API to define fast inner-loops.  You must provide a
 * strided loop.  If this is a cast between two "versions" of the same dtype
 * you must also provide an unaligned strided loop.
 * Other loops are useful to optimize the very common contiguous case.
 */
#define NPY_METH_strided_loop 3
#define NPY_METH_contiguous_loop 4
#define NPY_METH_unaligned_strided_loop 5
#define NPY_METH_unaligned_contiguous_loop 6


typedef struct {
    PyObject *caller;  /* E.g. the original ufunc, may be NULL */
    PyObject *method;  /* The method "self". Currently an opaque object */

    /* Operand descriptors, filled in by resolve_descriptors */
    PyArray_Descr **descriptors;
    /* Structure may grow (this is harmless for DType authors) */
} PyArrayMethod_Context;

typedef int (PyArrayMethod_StridedLoop)(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata);



/*
 * *********************************************************************
 *                            DTYPE API
 * *********************************************************************
 */

// TODO: These slots probably still need some thought, and/or a way to "grow"?
typedef struct{
    PyTypeObject *typeobj;    /* type of python scalar or NULL */
    int flags;                /* flags, including parametric and abstract */
    /* NULL terminated cast definitions. Use NULL for the newly created DType */
    PyArrayMethod_Spec **casts;
    PyType_Slot *slots;
    /* Baseclass or NULL (will always subclass `np.dtype`) */
    PyTypeObject *baseclass;
    int basicsize;  /* If 0, inherited, otherwise extends PyArray_Descr */
} PyArrayDTypeMeta_Spec;


typedef PyObject* __dtypemeta_fromspec(PyArrayDTypeMeta_Spec *dtype_spec);
#define PyArrayDTypeMeta_FromSpec \
    ((__dtypemeta_fromspec *)(__experimental_dtype_api_table[1]))



/*
 * *************************************************************************
 *                              Initialization
 * *************************************************************************
 *
 * Import the experimental API, the version must match the one defined in
 * the header to ensure changes are taken into account. NumPy will further
 * runtime-check this.
 * You must call this function to use the symbols in this file.
 */
#define __EXPERIMENTAL_DTYPE_VERSION 0

static int
import_experimental_dtype_api(int version)
{
    if (version != __EXPERIMENTAL_DTYPE_VERSION) {
        PyErr_SetString(PyExc_RuntimeError,
                "DType API version %d did not match header version %d. Please "
                "update the import statement and check for API changes.");
        return -1;
    }
    if (__experimental_dtype_api_table != __uninitialized_table) {
        /* already imported. */
        return 0;
    }

    PyObject *multiarray = PyImport_ImportModule("numpy.core._multiarray_umath");
    if (multiarray == NULL) {
        return -1;
    }
    printf("fetching table!\n");
    PyObject *api = PyObject_CallMethod(multiarray,
        "_get_experimental_dtype_api", "i", version);
    printf("fetched table!\n");
    Py_DECREF(multiarray);
    if (api == NULL) {
        return -1;
    }
    __experimental_dtype_api_table = PyCapsule_GetPointer(api,
            "experimental_dtype_api_table");
    Py_DECREF(api);
    printf("exported table: %p\n", __experimental_dtype_api_table);
    printf("    and func: %p\n", __experimental_dtype_api_table[0]);
    if (__experimental_dtype_api_table == NULL) {
        return -1;
    }
    return 0;
}

#endif  /* _NPY_EXPERIMENTAL_DTYPE_API_H */
