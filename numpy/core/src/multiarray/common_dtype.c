#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "common_dtype.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"


/*
 * This file defines all logic necessary for generic "common dtype"
 * operations.  This is unfortunately surprisingly complicated to get right
 * due to the value based logic NumPy uses and the fact that NumPy has
 * no clear (non-transitive) type promotion hierarchy.
 * Unlike most languages `int32 + float2 -> float64` instead of `float2`.
 * The other complicated thing is value-based-promotion, which means that
 * in many cases a Python 1, may end up as an `int8` or `uint8`.
 *
 * This file implements the necessary logic so that `np.result_type(...)`
 * can give the correct result for any order of inputs and can further
 * generalize to user DTypes.
 */


/**
 * This function defines the common DType operator.
 *
 * Note that the common DType will not be "object" (unless one of the dtypes
 * is object), even though object can technically represent all values
 * correctly.
 *
 * TODO: Before exposure, we should review the return value (e.g. no error
 *       when no common DType is found).
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return The common DType or NULL with an error set
 */
NPY_NO_EXPORT NPY_INLINE PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    common_dtype = dtype1->common_dtype(dtype1, dtype2);
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = dtype2->common_dtype(dtype2, dtype1);
    }
    if (common_dtype == NULL) {
        return NULL;
    }
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        PyErr_Format(PyExc_TypeError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
}


#define SWAP(x, y) do { \
        void *tmp = x;  \
        x = y;          \
        y = tmp;        \
    } while (0)


/**
 * This function takes a list of dtypes and "reduces" them (in a sense,
 * it finds the maximal dtype). Note that "maximum" here is defined by
 * knowledge (or category or domain). A user DType must always "know"
 * about all NumPy dtypes, floats "know" about integers, integers "know"
 * about unsigned integers.
 *
 *           c
 *          / \
 *         a   \    <-- The actual promote(a, b) may be c or unknown.
 *        / \   \
 *       a   b   c
 *
 * The reduction is done "pairwise". In the above `a.__common_dtype__(b)`
 * has a result (so `a` knows more) and `a.__common_dtype__(c)` returns
 * NotImplemented (so `c` knows more).  You may notice that the result
 * `res = a.__common_dtype__(b)` is not important.  We could try to use it
 * to remove the whole branch if `res is c` or by checking if
 * `c.__common_dtype(res) is c`.
 * Right now, we only clear initial elements in the most simple case where
 * `a.__common_dtype(b) is a` (and thus `b` cannot alter the end-result).
 * Clearing means, we do not have to worry about them later.
 *
 * There is one further subtlety. If we have an abstract DType and a
 * non-abstract one, we "prioritize" the non-abstract DType here.
 * In this sense "prioritizing" means that we use:
 *       abstract.__common_dtype__(other)
 * If both return NotImplemented (which is acceptable and even expected in
 * this case, see later) then `other` will be considered to know more.
 *
 * The reason why this may be acceptable for abstract DTypes, is that
 * the value-dependent abstract DTypes may provide default fall-backs.
 * The priority inversion effectively means that abstract DTypes are ordered
 * just below their concrete counterparts.
 * (This fall-back is convenient but not perfect, it can lead to
 * non-minimal promotions: e.g. `np.uint24 + 2**20 -> int32`. And such
 * cases may also be possible in some mixed type scenarios; they can be
 * avoided by defining the promotion explicitly in the user DType.)
 *
 * @param length Number of dtypes (and values)
 * @param dtypes
 * @param values Values for each DType, must be passed but is only used to
 *        ensure that the reordering applied to dtypes is also applied to
 *        values.
 */
static PyArray_DTypeMeta *
reduce_dtypes_to_most_knowledgeable(
        npy_intp length, PyArray_DTypeMeta **dtypes, PyObject **values)
{
    assert(length >= 2);
    npy_intp half = length / 2;

    PyArray_DTypeMeta *res = NULL;

    for (npy_intp low = 0; low < half; low++) {
        npy_intp high = length - low;
        if (dtypes[high]->abstract) {
            /*
             * Reverse priority: abstract DTypes at the same category are
             * considered lower. In particular, it allows the abstract DTypes
             * with a value to provide a fallback (and de-facto all logic).
             */
            npy_intp tmp = low;
            low = high;
            high = tmp;
        }

        if (dtypes[high] == dtypes[low]) {
            res = dtypes[low];
            Py_INCREF(res);
        }
        else {
            Py_XSETREF(res, dtypes[low]->common_dtype(dtypes[low], dtypes[high]));
            if (res == NULL) {
                return NULL;
            }
        }
        if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            SWAP(dtypes[low], dtypes[high]);
            SWAP(values[low], values[high]);
        }
        else if (res->abstract) {
            Py_INCREF(Py_NotImplemented);
            Py_SETREF(res, (PyArray_DTypeMeta *)Py_NotImplemented);
        }
        else if (res == dtypes[low]) {
            Py_SETREF(dtypes[high], NULL);
        }
    }

    if (length == 2) {
        return res;
    }
    Py_DECREF(res);
    return reduce_dtypes_to_most_knowledgeable(length - half, dtypes, values);
}

#undef SWAP

/*
 * Helper function to reduce a single DType in case it is abstract.
 * It will either discover from pyobject or use the default descriptor.
 * (In principle we could replace this with specialized functions to not
 * create a descriptor when we only need the class.)
 */
static NPY_INLINE PyArray_DTypeMeta *
convert_value_dependent_dtype(PyArray_DTypeMeta *dtype, PyObject *value)
{
    if (NPY_LIKELY(!dtype->abstract)) {
        Py_INCREF(dtype);
        return dtype;
    }
    PyArray_Descr *descr;
    if (value == NULL) {
        descr = dtype->default_descr(dtype);
    }
    else {
        descr = dtype->discover_descr_from_pyobject(dtype, value);
    }
    if (descr == NULL) {
        return NULL;
    }
    PyArray_DTypeMeta *res = NPY_DTYPE(descr);
    Py_DECREF(descr);
    return res;
}


/*
 * Finds the correct DTypes while honoring original NumPy dtypes and using
 * the default version otherwise.
 * The "future" version of this will probably have to
 */
static PyArray_DTypeMeta *
reduce_all_numpy_scalar_dtypes(
        npy_intp length, PyArray_DTypeMeta *dtypes[], PyObject *values[])
{
    PyArray_DTypeMeta *result = NULL;

    npy_intp new_length = 0;
    for (npy_intp i=0; i < length; i++) {
        if (values[i] != NULL) {
            dtypes[new_length] = convert_value_dependent_dtype(
                    dtypes[i], values[i]);
            if (dtypes[new_length] == NULL) {
                goto finish;
            }
            new_length += 1;
        }
    }
    result = PyArray_PromoteDTypesWithValues(new_length, dtypes, NULL);

  finish:
    /* We replaced the DTypes, so they are not borrowed anymore */
    for (npy_intp i=0; i < new_length; i++) {
        Py_DECREF(dtypes[i]);
    }
  return result;
}


/*
 * When dealing with potential values, we need to find the "minimum scalar
 * type" with the legacy way of doing this.
 * We would also like to change this in the future (as of NumPy 1.21), whic
 * may require warnings.  So the way this is implemented is by first
 * reducing _without_ value.
 */
static PyObject *
reduce_dtypes_with_value()
{

    int abstract_value_type = -1;
    int is_small_unsigned = 1;  /* If true */
    npy_intp negative_integer_pos = -1;
    npy_intp remaining_length = 0;
    for (npy_intp i = 0; i < length; i++) {
        PyArray_DTypeMeta *dtype = dtypes[i];
        if (dtypes[i] == NULL) {
            continue;
        }
        Py_INCREF(dtypes[i]);
        Py_XINCREF(values[i]);
        if (i > remaining_length) {
            dtypes[remaining_length] = dtypes[i];
            values[remaining_length] = values[i];
        }
        remaining_length += 1;
    }
}


/**
 * Promotes a list of DTypes with each other in a way that should guarantee
 * stable results even when changing the order.
 *
 * In general this approach always works as long as the most generic dtype
 * is either strictly larger, or compatible with all other dtypes.
 * For example promoting float16 with any other float, integer, or unsigned
 * integer again gives a floating point number. And any floating point number
 * promotes in the "same way" as `float16`.
 * If a user inserts more than one type into the NumPy type hierarchy, this
 * can break. Given:
 *     uint24 + int32 -> int48  # Promotes to a *new* dtype!
 *
 * The following becomes problematic (order does not matter):
 *         uint24 +      int16  +           uint32  -> int64
 *    <==      (uint24 + int16) + (uint24 + uint32) -> int64
 *    <==                int32  +           uint32  -> int64
 *
 * It is impossible to achieve an `int48` result in the above.
 *
 * This is probably only resolvable by asking `uint24` to take over the
 * whole reduction step; which we currently do not do.
 * (It may be possible to notice the last up-cast and implement use something
 * like: `uint24.nextafter(int32).__common_dtype__(uint32)`, but that seems
 * even harder to grasp.)
 *
 * Note that a case where two dtypes are mixed (and know nothing about each
 * other) will always generate an error:
 *     uint24 + int48 + int64 -> Error
 *
 * Even though `int64` is a safe solution, since `uint24 + int64 -> int64` and
 * `int48 + int64 -> int64` and `int64` and there cannot be a smaller solution.
 *
 * //TODO: Maybe this function should allow not setting an error?
 *
 * @param length Number of dtypes (and values) must be at least 1
 * @param dtypes The concrete or abstract DTypes to promote
 * @param values NULL or a list of values for each DType (entries may be NULL
 *        and are expected to be for concrete DTypes or abstract ones that are
 *        not value dependent.)
 * @return NULL or the promoted DType.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypesWithValues(
        npy_intp length, PyArray_DTypeMeta **dtypes_in, PyObject **values_in)
{
    assert(length >= 1);
    if (length == 1) {
        PyObject *value = values_in == NULL ? NULL : values_in[0];
        return convert_value_dependent_dtype(dtypes_in[0], value);
    }
    PyArray_DTypeMeta *result = NULL;

    /*
     * Copy both dtypes and values so that we can reorder them.
     */
    PyObject *_scratch_stack[NPY_MAXARGS * 2];
    PyObject **_scratch_heap = NULL;
    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **)_scratch_stack;
    PyObject **values = _scratch_heap + NPY_MAXARGS;

    if (length > NPY_MAXARGS) {
        _scratch_heap = PyMem_Malloc(length * 2 * sizeof(PyObject *));
        if (_scratch_heap == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        dtypes = (PyArray_DTypeMeta **)_scratch_heap;
        values = _scratch_heap + length;
    }

    memcpy(dtypes, dtypes_in, length * sizeof(PyObject *));
    if (values == NULL) {
        memset(values, 0, length * sizeof(PyObject *));
    }
    else {
        memcpy(values, values_in, length * sizeof(PyObject *));
    }

    /*
     * `result` is the last promotion result, which can usually be reused.
     * The passed in dtypes and values are partially sorted (and potentially
     * cleared). `dtypes[0]` will be the most knowledgeable (highest category).
     */
    result = reduce_dtypes_to_most_knowledgeable(length, dtypes, values);
    if (result == NULL) {
        goto error;
    }

    /*
     * The following step is important for dealing with value based abstract
     * DTypes (specifically python integers, complex, and floats).
     * A potential solution to many of these things, is to allow implementation
     * of a `result.__common_dtype_reduce__(*dtypes)`, however, certain
     * behaviour is very complex and we do not want to expose much of it to
     * user DTypes (especially not initially).
     *
     * For simplicity we remove NULLs from dtypes (and values) and INCREF
     * both (as necessary).
     *
     * There are two further pre-processing steps here:
     *
     * 1. For user DTypes (non legacy), we assume that value based casting
     *    should not be a thing to begin with. We discard all values in this
     *    case and similar to 1 we honor the original NumPy DType when
     *    available. However, unlike point 1, we do not replace the abstract
     *    DType with its default.
     *    The user DType is expected to define a meaningful promotion with the
     *    abstract DType (but no value), or an error will occur.
     *    NOTE: We could provide a default (i.e. float64, long).
     *
     * 2. If (and only if) the DType we ended up with at the end is one
     *    of our "Python scalar" abstract DTypes, we need to convert all
     *    of them to their "default" (it must be only ours, otherwise we would
     *    have ended up with a user DType). This means we honor the original
     *    NumPy DType (if available).
     *
     * 3. In the "normal" case, we do two things: By now, we know that we
     *    (potentially) use current NumPy value based logic. So we handle it
     *    right here and now.
     *    That is find the minimal (unsigned) integer or (complex) float
     *    precision.  This could be moved into either a reduce like logic,
     *    or potentially a `dtype.__common_dtype_with_value__()` method
     *    as well. But at long we are OK with using point 2 above for user
     *    DTypes (and as a potential future here), this seems easier.
     *
     * This currently hard codes the three cases above, instead of trying to
     * generalize them more broadly, which also locks out user DTypes from
     * all value based promotion (as opposed to "weak" promotion).
     * Aside from implementing the reduction itself with
     * `dtype.__common_dtype_reduce__(*dtypes)`, removing these special cases
     * may require an asymmetry, i.e. a method `__rcommon_dtype__`
     * (possibly with values).  This could allow a "default" fallback for
     * value based promotion (point 2 and 3 generalization) defined on the
     * abstract DType itself.
     */

    /*
     * Before getting into the difficult cases, we first "recuce" the dtypes
     * using "weak" type logic for the "values".  In all cases, we will have
     * to deal with the values afterwards, unfortunately!
     */
    PyArray_DTypeMeta *main_dtype = dtypes[0];

    npy_intp reduce_start = 1;
    if (result == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_SETREF(result, NULL);
    }
    else {
        /* (new) first value is already taken care of in `result` */
        reduce_start = 2;
    }
    /*
     * NOTE: The code assumes that by doing the pairwise "maximum" reduction
     *       most, or all, corner cases are smoothed out.  It is certainly
     *       possible to design ways to break it, but it should be safe as
     *       long as there is a clear hierarchy of which dtype knows about
     *       which other dtype (i.e. if int24 > int96 and int96 > int48 then
     *       we assume that int24 > int48 -- i.e. int24 must also know about
     *       int48).  The main way this might break is potentially the
     *       "complex" abstract dtype (for python complex scalars), because
     *       its category is "inexact" rather than "complex".  However,
     *       the worst failure path here is probably that some orders error
     *       while others succeed with reasonable results).
     */
    PyArray_DTypeMeta *prev = NULL;
    for (npy_intp i = reduce_start; i < length; i++) {
        if (dtypes[i] == NULL || dtypes[i] == prev) {
            continue;
        }
        /*
         * "Promote" the current dtype with the main one (which should be
         * a higher category). We assume that the result is not in a lower
         * category.
         */
        PyArray_DTypeMeta *promotion = main_dtype->common_dtype(
                main_dtype, dtypes[i]);
        if (promotion == NULL) {
            goto error;
        }
        if (result == NULL) {
            result = promotion;
            continue;
        }

        /*
         * The above promoted, now "reduce" with the current result; note that
         * in the typical cases we expect this step to be a no-op.
         */
        Py_SETREF(result, PyArray_CommonDType(result, promotion));
        Py_DECREF(promotion);
        if (result == NULL) {
            goto error;
        }
    }

    /*
     * To repeat the above, we still have to deal with three cases here:
     *
     * 1. User DTypes, should not use value-based logic for 0-D arrays, but
     *    did until now. So they need promotion with the "existing dtype".
     * 2. Our dtypes, must use value-based (as opposed to "weak" that was
     *    used until here).  However, they should transition to what user
     *    dtypes do.
     *    At this time, they need to apply value-based logic to upcast,
     *    in the future, they have to do the same as user DTypes as well and
     *    then compare it and warn.
     * 3.
     */
    if (result == &PyArray_PyIntAbstractDType ||
            result == &PyArray_PyFloatAbstractDType ||
            result == &PyArray_PyComplexAbstractDType) {
        /*
         * Case 3 -- We ended up with our abstract DType (i.e. value) and it
         * is always necessary to convert all those to their "default".
         */
        Py_SETREF(result, reduce_all_numpy_scalar_dtypes(length, dtypes, values));
        goto finish;
    }
    else if (result->type_num < NPY_NTYPES) {
        /*
         * Case 2 -- NumPy style value-based promotion.
         */
        Py_SETREF(result, reduce_dtypes_with_value(result, dtypes, values));
        goto finish;
    }
    else {
        /*
         * Case 3 -- User dtypes, this could be an abstract dtype, but let
         * that pass at this point.  It should not requires the original value.
         *
         * NOTE: This path is important because eventually we will have to
         *       transition is to honor the original NumPy dtypes for 0-D array
         *       inputs. (But not in *all* cases)
         */
    }

    goto finish;

  error:
    Py_XSETREF(result, NULL);

  finish:
    PyObject_Free(_scratch_heap);
    return result;
}
