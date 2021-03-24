#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "common.h"

#include "experimental_public_dtype_api.h"
#include "array_method.h"


#define EXPERIMENTAL_DTYPE_API_VERSION 0


#define NPY_DTYPE_PARAMETRIC 1
#define NPY_DTYPE_ABSTRACT 2

typedef struct{
    char *name;
    char *module;
    PyTypeObject *typeobj;    /* type of python scalar or NULL */
    int flags;                /* flags, including parametric and abstract */
    /* NULL terminated cast definitions. Use NULL for the newly created DType */
    PyArrayMethod_Spec **casts;
    PyType_Slot *slots;
    /* Baseclass or NULL (will always subclass `np.dtype`) */
    PyTypeObject *baseclass;
} PyArrayDTypeMeta_Spec;


#define NPY_DT_discover_descr_from_pyobject 1
#define _NPY_DT_is_known_scalar_type 2
#define NPY_DT_default_descr 3
#define NPY_DT_common_dtype 4
#define NPY_DT_common_instance 5
#define NPY_DT_setitem 6
#define NPY_DT_getitem 7


static PyObject *
PyArrayDTypeMeta_FromSpec(PyArrayDTypeMeta_Spec *spec)
{
    if (spec->flags & ~(NPY_DTYPE_ABSTRACT|NPY_DTYPE_PARAMETRIC)) {
        PyErr_SetString(PyExc_TypeError,
                "unknown DType flags passed.");
        return NULL;
    }
    if (spec->typeobj == NULL) {
        PyErr_SetString(PyExc_TypeError,
                "Not giving a type object is currently not supported, but "
                "is expected to be supported eventually.  This would mean "
                "that e.g. indexing a NumPy array will return a 0-D array "
                "and not a scalar.");
        return NULL;
    }

    /*
     * This is somewhat horrible. But use the Python API to create the new
     * type.  The reason for this is that we are creating a new extension
     * metaclass... Something that is completely unproblematic in most ways
     * and completely unsupported in the limited API.
     * Since, I do not want to expose the DTypeMeta struct, there is a
     * conundrum, internally, we have to use the full API for defining the
     * DTypeMeta struct, but at the same time, NumPy has strict ABI
     * limitations, and it would be nice to not force this on the user.
     *
     * (We could expose it to the user, since we could extend the metaclass
     * by a single pointer, which we then use to store our functionality
     * while keeping the "slots" behind the pointer fully opaque.)
     */
    PyObject *bases;
    if (spec->baseclass == NULL) {
        bases = PyTuple_Pack(1, &PyArrayDescr_Type);
    }
    else {
        bases = PyTuple_Pack(2, spec->baseclass, &PyArrayDescr_Type);
    }
    if (bases == NULL) {
        return NULL;
    }
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        Py_DECREF(bases);
        return NULL;
    }
    PyObject *args = Py_BuildValue("sNN", spec->name, bases, dict);
    if (args == NULL) {
        return NULL;
    }
    PyArray_DTypeMeta *DType = (PyArray_DTypeMeta *)PyType_Type.tp_new(
            &PyArrayDTypeMeta_Type, args, NULL);
    if (DType == NULL) {
        Py_DECREF(args);
        return NULL;
    }
    /* I guess init is a no-op here? But lets do it... */
    int res = PyType_Type.tp_init((PyObject *)DType, args, NULL);
    Py_DECREF(args);
    if (res < 0) {
        return NULL;
    }

    /*
     * OK, now that we are done with the horrible, continue with the weird :).
     */
    PyType_Slot *spec_slot = spec->slots;

    while (1) {
        int slot = spec_slot->slot;
        void *pfunc = spec_slot->pfunc;
        spec_slot++;
        if (slot == 0) {
            break;
        }

        if (slot == Npy_dt_) {

        }
        else if (slot == Npy_dt)

    }

    return (PyObject *)DType;
}












static void *experimental_api_table[] = {
        &PyArrayMethod_FromSpec,
        &PyArrayDTypeMeta_FromSpec,
        NULL,
};


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

    return PyCapsule_New(&experimental_api_table,
            "experimental_dtype_api_table", NULL);
}
