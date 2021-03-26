#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include "numpy/arrayobject.h"
#include "common.h"

#include "experimental_public_dtype_api.h"
#include "array_method.h"
#include "dtypemeta.h"
#include "array_coercion.h"
#include "convert_datatype.h"


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



static PyArray_DTypeMeta *
dtype_does_not_promote(
        PyArray_DTypeMeta *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(other))
{
    /* `other` is guaranteed not to be `self`, so we don't have to do much... */
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


static PyArray_Descr *
discover_as_default(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return cls->default_descr(cls);
}


static PyArray_Descr *
use_new_as_default(PyArray_DTypeMeta *self)
{
    PyObject *res = PyObject_CallNoArgs((PyObject *)self);
    if (res == NULL) {
        return NULL;
    }
    /*
     * Lets not trust that the DType is implemented correctly
     * TODO: Should probably do an exact type-check (at least unless this is
     *       an abstract DType).
     */
    if (!PyArray_DescrCheck(res)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Instantiating %S did not return a dtype instance, this is "
                "invalid (especially without a custom `default_descr()`).");
        Py_DECREF(res);
        return NULL;
    }
    PyArray_Descr *descr = (PyArray_Descr *)res;
    /*
     * Should probably do some more sanity checks here on the descriptor
     * to ensure the user is not being naughty. But in the end, we have
     * only limited control anyway.
     */
    return descr;
}


typedef int(setitemfunction)(PyArray_Descr *, PyObject *, char *);
typedef PyObject *(getitemfunction)(PyArray_Descr *, char *);


static int
legacy_setitem_using_DType(PyObject *obj, void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return -1;
    }
    setitemfunction *setitem = NPY_DTYPE(PyArray_DESCR(arr))->setitem;
    return setitem(PyArray_DESCR(arr), obj, data);
}


static PyObject *
legacy_getitem_using_DType(void *data, void *arr)
{
    if (arr == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Using legacy SETITEM with NULL array object is only "
                "supported for basic NumPy DTypes.");
        return NULL;
    }
    getitemfunction *getitem = NPY_DTYPE(PyArray_DESCR(arr))->getitem;
    return getitem(PyArray_DESCR(arr), data);
}



PyArray_ArrFuncs default_funcs = {
        .setitem = &legacy_setitem_using_DType,
        .getitem = &legacy_getitem_using_DType
};


/* other slots are in order, so keep only last around: */
#define NUM_DTYPE_SLOTS 7


static PyObject *
PyArrayDTypeMeta_FromSpec(PyArrayDTypeMeta_Spec *spec)
{
    if (spec->flags & ~(NPY_DTYPE_ABSTRACT|NPY_DTYPE_PARAMETRIC)) {
        PyErr_SetString(PyExc_TypeError,
                "unknown DType flags passed.");
        return NULL;
    }
    if (spec->typeobj == NULL || !PyType_Check(spec->typeobj)) {
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
        Py_DECREF(DType);
        return NULL;
    }

    /*
     * OK, now that we dealt with the horrible, continue with the slots :).
     */
    /* Set default values (where applicable) */
    DType->discover_descr_from_pyobject = &discover_as_default;
    DType->is_known_scalar_type = &python_builtins_are_known_scalar_types;
    DType->default_descr = use_new_as_default;
    DType->common_dtype = dtype_does_not_promote;
    DType->common_instance = NULL;  /* May need a default for non-parametric? */
    DType->setitem = NULL;
    DType->getitem = NULL;

    PyType_Slot *spec_slot = spec->slots;
    while (1) {
        int slot = spec_slot->slot;
        void *pfunc = spec_slot->pfunc;
        spec_slot++;
        if (slot == 0) {
            break;
        }
        if (slot > NUM_DTYPE_SLOTS || slot < 0) {
            PyErr_Format(PyExc_RuntimeError,
                    "Invalid slot with value %d passed in.", slot);
            Py_DECREF(DType);
            return NULL;
        }
        /*
         * It is up to the user to get this right, and slots are sorted
         * exactly like they are stored right now:
         */
        void **current = (void **)(&(DType->discover_descr_from_pyobject));
        current += slot - 1;
        *current = pfunc;
    }
    if (DType->setitem == NULL || DType->getitem == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "A DType must provide a getitem/setitem (there may be an "
                "exception here in the future if no scalar type is provided)");
        Py_DECREF(DType);
        return NULL;
    }

    if (spec->flags & NPY_DTYPE_ABSTRACT) {
        DType->abstract = 1;
    }
    if (spec->flags & NPY_DTYPE_PARAMETRIC) {
        DType->parametric = 1;
        if (DType->common_instance == NULL ||
                DType->discover_descr_from_pyobject == &discover_as_default) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Parametric DType must define a common-instance and "
                    "descriptor discovery function!");
            return NULL;
        }
    }
    if (spec->flags & ~(NPY_DTYPE_ABSTRACT | NPY_DTYPE_PARAMETRIC)) {
        PyErr_SetString(PyExc_RuntimeError,
                "Invalid DType flag set.");
        Py_DECREF(DType);
        return NULL;
    }
    DType->f = &default_funcs;
    /* invalid type num. Ideally, we get away with it! */
    DType->type_num = -1;

    /*
     * Handle the scalar type mapping.
     */
    Py_INCREF(spec->typeobj);
    DType->scalar_type = spec->typeobj;
    if (PyType_GetFlags(spec->typeobj) & Py_TPFLAGS_HEAPTYPE) {
        if (PyObject_SetAttrString((PyObject *)DType->scalar_type,
                "__associated_array_dtype__", (PyObject *)DType) < 0) {
            Py_DECREF(DType);
            return NULL;
        }
    }
    if (_PyArray_MapPyTypeToDType(DType, DType->scalar_type, 0) < 0) {
        Py_DECREF(DType);
        return NULL;
    }

    /* Ensure cast dict is defined (not sure we have to do it here) */
    DType->castingimpls = PyDict_New();
    if (DType->castingimpls == NULL) {
        Py_DECREF(DType);
        return NULL;
    }
    /*
     * And now, register all the casts that are currently defined!
     */
    PyArrayMethod_Spec **next_meth_spec = spec->casts;
    while (1) {
        PyArrayMethod_Spec *meth_spec = *next_meth_spec;
        next_meth_spec++;
        if (meth_spec == NULL) {
            break;
        }
        /*
         * The user doesn't know the name of DType yet, so we have to fill it
         * in for them!
         */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == NULL) {
                meth_spec->dtypes[i] = DType;
            }
        }
        /* Register the cast! */
        res = PyArray_AddCastingImplementation_FromSpec(meth_spec, 0);

        /* Also clean up again, so nobody can get bad ideas... */
        for (int i=0; i < meth_spec->nin + meth_spec->nout; i++) {
            if (meth_spec->dtypes[i] == DType) {
                meth_spec->dtypes[i] = NULL;
            }
        }

        if (res < 0) {
            Py_DECREF(DType);
            return NULL;
        }
    }

    if (DType->within_dtype_castingimpl == NULL) {
        /*
         * We expect this for now. We should have a default for DType that
         * only support simple copy (and possibly byte-order when assuming that
         * they swap the full itemsize).
         */
        PyErr_SetString(PyExc_RuntimeError,
                "DType must provide a function to cast (or just copy) between "
                "its own instances!");
        Py_DECREF(DType);
        return NULL;
    }

    /* And finally, we have to register all the casts! */
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
