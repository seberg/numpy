#include <Python.h>
#include "pyerrors.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "abstractdtypes.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "common_dtype.h"
#include "convert_datatype.h"

#include "special_comparisons.h"

extern "C" {
    #include "loops.h"
}

/*
 * Helper for templating, avoids warnings about uncovered switch paths.
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);
            return nullptr;
    }
}


template <COMP comp>
static int
normal_comparison_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /* old loops are missing const, so cast it away: */
    char **args = (char **)data;
    switch (comp) {
        case COMP::EQ:
            ULONGLONG_equal(args, dimensions, strides, NULL);
            break;
        case COMP::NE:
            ULONGLONG_not_equal(args, dimensions, strides, NULL);
            break;
        case COMP::LT:
            ULONGLONG_less(args, dimensions, strides, NULL);
            break;
        case COMP::LE:
            ULONGLONG_less_equal(args, dimensions, strides, NULL);
            break;
        case COMP::GT:
            ULONGLONG_greater(args, dimensions, strides, NULL);
            break;
        case COMP::GE:
            ULONGLONG_greater_equal(args, dimensions, strides, NULL);
            break;
    }
    return 0;
}


template <bool result>
static int
fixed_result_loop(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *out = data[2];
    npy_intp stride = strides[2];

    while (N--) {
        *reinterpret_cast<npy_bool *>(out) = result;
        out += stride;
    }
    return 0;
}


static NPY_CASTING
resolve_descriptors_raw(
    PyArrayMethodObject *self, PyArray_DTypeMeta **dtypes,
    PyArray_Descr **given_descrs, PyObject **input_scalars,
    PyArray_Descr **loop_descrs, npy_intp *view_offset)
{
    int value_range = 0;  /* -1, 0, or 1  (<0, fits, >MAX_ULONGLONG) */

    if (input_scalars[1] != NULL && PyLong_CheckExact(input_scalars[1])) {
        int overflow;

        long long val = PyLong_AsLongLongAndOverflow(input_scalars[1], &overflow);
        if (val == -1 && PyErr_Occurred()) {
            return (NPY_CASTING)-1;  /* should not be possible */
        }
        if (overflow == 0) {
            value_range = val < 0 ? -1 : 0;
        }
        else if (overflow < 0) {
            value_range = -1;
        }
        else {
            unsigned long long val = PyLong_AsUnsignedLongLong(input_scalars[1]);
            if (val == (unsigned long long)-1 && PyErr_Occurred()) {
                PyErr_Clear();  /* Seems fair to assume it is overflow */
                value_range = 1;
            }
            else {
                value_range = 0;
            }
        }
    }
    if (value_range == 0) {
        Py_INCREF(dtypes[0]->singleton);
        loop_descrs[1] = dtypes[0]->singleton;
    }
    else if (value_range < 0) {
        /* Using the singleton here object dtype here (see below) */
        loop_descrs[1] = PyArray_DescrFromType(NPY_OBJECT);
    }
    else {
        /* 
         * HACK: Create a *new* dtype to indicate this is a huge value!
         */
        loop_descrs[1] = PyArray_DescrNewFromType(NPY_OBJECT);
        if (loop_descrs[1] == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    Py_INCREF(dtypes[0]->singleton);
    loop_descrs[0] = dtypes[0]->singleton;
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);

    return NPY_NO_CASTING;
}


template<COMP comp>
static int
get_loop(PyArrayMethod_Context *context,
        int aligned, int move_references, const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    if (context->descriptors[1]->type_num == context->descriptors[0]->type_num) {
        *out_loop = &normal_comparison_loop<comp>;
    }
    else {
        assert(context->descriptors[1]->type_num == NPY_OBJECT);
        /* HACK: If the descr is the singleton the result is smaller */
        PyArray_Descr *obj_singleton = PyArray_DescrFromType(NPY_OBJECT);
        if (context->descriptors[1] == obj_singleton) {
            /* Second argument must be smaller, the result is trivial: */
            switch (comp) {
                case COMP::EQ:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        else {
            /* Second argument must larger, the result is trivial: */
            switch (comp) {
                case COMP::EQ:
                case COMP::GT:
                case COMP::GE:
                    *out_loop = &fixed_result_loop<false>;
                    break;
                case COMP::NE:
                case COMP::LT:
                case COMP::LE:
                    *out_loop = &fixed_result_loop<true>;
                    break;
            }
        }
        Py_DECREF(obj_singleton);
    }
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    return 0;
}


/*
 * Machinery to add the string loops to the existing ufuncs.
 */

/*
 * This function replaces the strided loop with the passed in one,
 * and registers it with the given ufunc.
 */
static int
add_loop(PyObject *umath, const char *ufunc_name,
         PyArrayMethod_Spec *spec, void *loop)
{
    PyObject *name = PyUnicode_FromString(ufunc_name);
    if (name == nullptr) {
        return -1;
    }
    PyObject *ufunc = PyObject_GetItem(umath, name);
    Py_DECREF(name);
    if (ufunc == nullptr) {
        return -1;
    }
    spec->slots[0].pfunc = (void *)loop;

    int res = PyUFunc_AddLoopFromSpec(ufunc, spec);
    Py_DECREF(ufunc);
    return res;
}


template<COMP...>
struct add_loops;

template<>
struct add_loops<> {
    int operator()(PyObject*, PyArrayMethod_Spec*) {
        return 0;
    }
};

template<COMP comp>
static void *
get_get_loop()
{
    return (void *)get_loop<comp>;
}

template<COMP comp, COMP... comps>
struct add_loops<comp, comps...> {
    int operator()(PyObject* umath, PyArrayMethod_Spec* spec) {
        void* loop = get_get_loop<comp>();

        if (add_loop(umath, comp_name(comp), spec, loop) < 0) {
            return -1;
        }
        else {
            return add_loops<comps...>()(umath, spec);
        }
    }
};


NPY_NO_EXPORT int
init_special_int_comparisons(PyObject *umath)
{
    int res = -1;
    /* NOTE: This should receive global symbols? */
    PyArray_DTypeMeta *UInt = PyArray_DTypeFromTypeNum(NPY_UINT64);
    PyArray_DTypeMeta *PyInt = &PyArray_PyIntAbstractDType;
    PyArray_DTypeMeta *Bool = PyArray_DTypeFromTypeNum(NPY_BOOL);

    /* We start with the string loops: */
    PyArray_DTypeMeta *dtypes[] = {UInt, PyInt, Bool};
    /*
     * We only have one loop right now, the strided one.  The default type
     * resolver ensures native byte order/canonical representation.
     */
    PyType_Slot slots[] = {
        {_NPY_METH_get_loop, nullptr},
        {NPY_METH_resolve_descriptors_raw, (void *)&resolve_descriptors_raw},
        {0, NULL},
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_pyint_comp";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* All String loops */
    using comp_looper = add_loops<COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (comp_looper()(umath, &spec) < 0) {
        goto finish;
    }

    res = 0;
  finish:
    Py_DECREF(UInt);
    Py_DECREF(Bool);
    return res;
}
