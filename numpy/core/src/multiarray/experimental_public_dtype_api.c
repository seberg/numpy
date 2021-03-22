#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "common.h"

#include "experimental_public_dtype_api.h"
#include "array_method.h"


#define EXPERIMENTAL_DTYPE_API_VERSION 0

static void *experimental_api_table[2];


NPY_NO_EXPORT PyObject *
_get_experimental_dtype_api(PyObject *NPY_UNUSED(mod), PyObject *arg)
{
    char *env = getenv("NUMPY_EXPERIMENTAL_DTYPE_API");
    if (env == NULL || strcmp(env, "1") != 0) {
        PyErr_Format(PyExc_RuntimeError,
                "The new DType API is currently in an exploratory phase and "
                "should NOT be used for production code.  "
                "Expect modifications and crashes!  "
                "To experiment with the new API you must set "
                "`NUMPY_EXPERIMENTAL_DTYPE_API=1` as an environment variable.");
        return NULL;
    }

    long version = PyLong_AsLong(arg);
    if (error_converting(version)) {
        return NULL;
    }
    if (version != EXPERIMENTAL_DTYPE_API_VERSION) {
        PyErr_Format(PyExc_RuntimeError,
                "Experimental DType API version %d requested, but NumPy "
                "is exporting version %d.  If your version is lower, please "
                "recompile.  If your version is higher, upgrade NumPy.",
                version, EXPERIMENTAL_DTYPE_API_VERSION);
        return NULL;
    }

    experimental_api_table[0] = &PyArrayMethod_FromSpec;
    printf("PyArrayMethod_FromSpec: %p\n", &PyArrayMethod_FromSpec);

    return PyCapsule_New(&experimental_api_table,
            "experimental_dtype_api_table", NULL);
}
