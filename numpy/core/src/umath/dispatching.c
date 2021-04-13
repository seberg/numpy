/*
 * This file implements universal function dispatching and promotion (which
 * is necessary to happen before dispatching).
 * As such it works on the UFunc object.
 */

#include "dispatching.h"
#include "dtypemeta.h"
#include "npy_hashtable.h"


/**
 * Resolves the implementation to use, this uses typical multiple dispatching
 * methods of finding the best matching implementation or resolver.
 * (Based on `isinstance()`, the knowledge that non-abstract DTypes cannot
 * be subclassed is used, however.)
 *
 * @param ufunc
 * @param op_dtypes The DTypes that are either passed in (defined by an
 *        operand) or defined by the `signature` as also passed in as
 *        `fixed_DTypes`.
 * @param out_info Returns the tuple describing the best implementation
 *        (consisting of dtypes and ArrayMethod or promoter).
 *        WARNING: Returns a borrowed reference!
 * @returns -1 on error 0 on success.  Note that the output can be NULL on
 *          success if nothing is found.
 */
static int
resolve_implementation_info(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyObject **out_info)
{
    int nargs = ufunc->nargs;
    /* Use new style type resolution has to happen... */
    Py_ssize_t size = PySequence_Length(ufunc->_loops);
    PyObject *best_dtypes = NULL;
    PyObject *best_resolver_info = NULL;

    for (Py_ssize_t res_idx = 0; res_idx < size; res_idx++) {
        /* Test all resolvers  */
        PyObject *resolver_info = PySequence_Fast_GET_ITEM(
                ufunc->_loops, res_idx);
        PyObject *curr_dtypes = PyTuple_GET_ITEM(resolver_info, 0);

        // TODO: If we have abstract, there is no point in checking
        //       a UFuncImpl (which is always concrete)
        /*
         * Test if the current resolver matches, it could make sense to
         * reorder these checks to avoid the IsSubclass check as much as
         * possible.
         */
#if NPY_UF_DBG_TRACING
        printf("Check if the resolver/loop matches:\n");
        PyObject_Print(dtype_tuple, stdout, 0);
        PyObject_Print(curr_dtypes, stdout, 0);
        printf("\n");
#endif
        npy_bool matches = NPY_TRUE;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            PyArray_DTypeMeta *given_dtype = op_dtypes[i];
            PyArray_DTypeMeta *resolver_dtype = (
                    (PyArray_DTypeMeta *)PyTuple_GET_ITEM(curr_dtypes, i));
            if (given_dtype == (PyArray_DTypeMeta *)Py_None) {
                /* If None is given, anything will match. */
                continue;
            }
            if (given_dtype == resolver_dtype) {
                continue;
            }
            if (!resolver_dtype->abstract) {
                matches = NPY_FALSE;
                break;
            }
            int subclass = PyObject_IsSubclass(
                    (PyObject *)given_dtype, (PyObject *)resolver_dtype);
            if (subclass < 0) {
                return -1;
            }
            if (!subclass) {
                matches = NPY_FALSE;
                break;
            }
        }
        if (!matches) {
            continue;
        }

#if NPY_UF_DBG_TRACING
        printf("    Found a match!\n");
#endif
        /* The resolver matches, but we have to check if it is better */
        if (best_dtypes != NULL) {
            int current_best = -1;  /* -1 neither, 0 current best, 1 new */
            /*
             * If both have concrete and None in the same position and
             * they are identical, we will continue searching using the
             * first best for comparison, in an attempt to find a better
             * one.
             * In all cases, we give up resolution, since it would be
             * necessary to compare to two "best" cases.
             */
            int unambiguous_equivally_good = 1;
            for (Py_ssize_t i = 0; i < nargs; i++) {
                int best;

                /* Whether this (output) dtype was specified at all */
                int is_not_specified = (
                        op_dtypes[i] == (PyArray_DTypeMeta *)Py_None);

                PyObject *prev_dtype = PyTuple_GET_ITEM(best_dtypes, i);
                PyObject *new_dtype = PyTuple_GET_ITEM(curr_dtypes, i);

                if (prev_dtype == new_dtype) {
                    /* equivalent, so this entry does not matter */
                    continue;
                }
                if (is_not_specified) {
                    /*
                     * When DType is completely unspecified, prefer abstract
                     * over concrete, assuming it will resolve.
                     * Furthermore, we cannot decide which abstract/None
                     * is "better", only concrete ones which are subclasses
                     * of Abstract ones are defined as worse.
                     */
                    int prev_is_concrete = 0, new_is_concrete = 0;
                    if ((prev_dtype != Py_None) &&
                        (!((PyArray_DTypeMeta *)prev_dtype)->abstract)) {
                        prev_is_concrete = 1;
                    }
                    if ((new_dtype != Py_None) &&
                        (!((PyArray_DTypeMeta *)new_dtype)->abstract)) {
                        new_is_concrete = 1;
                    }
                    if (prev_is_concrete == new_is_concrete) {
                        best = -1;
                    }
                    else if (prev_is_concrete) {
                        unambiguous_equivally_good = 0;
                        best = 1;
                    }
                    else {
                        unambiguous_equivally_good = 0;
                        best = 0;
                    }
                }
                    /* If either is None, the other is strictly more specific */
                else if (prev_dtype == Py_None) {
                    unambiguous_equivally_good = 0;
                    best = 1;
                }
                else if (new_dtype == Py_None) {
                    unambiguous_equivally_good = 0;
                    best = 0;
                }
                    /*
                     * If both are concrete and not identical, this is
                     * ambiguous.
                     */
                else if (!((PyArray_DTypeMeta *)prev_dtype)->abstract &&
                         !((PyArray_DTypeMeta *)new_dtype)->abstract) {
                    /*
                     * Ambiguous unless the are identical (checked above),
                     * but since they are concrete it does not matter which
                     * best to compare.
                     */
                    best = -1;
                }
                else if (!((PyArray_DTypeMeta *)prev_dtype)->abstract) {
                    /* old is not abstract, so better (both not possible) */
                    unambiguous_equivally_good = 0;
                    best = 0;
                }
                else if (!((PyArray_DTypeMeta *)new_dtype)->abstract) {
                    /* new is not abstract, so better (both not possible) */
                    unambiguous_equivally_good = 0;
                    best = 1;
                }
                    /*
                     * Both are abstract DTypes, there is a clear order if
                     * one of them is a subclass of the other.
                     * If this fails, reject it completely (could be changed).
                     * The case that it is the same dtype is already caught.
                     */
                else {
                    /* Note the identity check above, so this true subclass */
                    int new_is_subclass = PyObject_IsSubclass(
                            new_dtype, prev_dtype);
                    if (new_is_subclass < 0) {
                        return -1;
                    }
                    /*
                     * Could optimize this away if above is True, but this
                     * catches inconsistent definitions of subclassing.
                     */
                    int prev_is_subclass = PyObject_IsSubclass(
                            prev_dtype, new_dtype);
                    if (prev_is_subclass < 0) {
                        return -1;
                    }
                    if (prev_is_subclass && new_is_subclass) {
                        /* should not happen unless they are identical */
                        PyErr_SetString(PyExc_RuntimeError,
                                "inconsistent subclassing of DTypes; if "
                                "this happens, two dtypes claim to be a "
                                "superclass of the other one.");
                        return -1;
                    }
                    if (!prev_is_subclass && !new_is_subclass) {
                        /* Neither is more precise than the other one */
                        PyErr_SetString(PyExc_TypeError,
                                "inconsistent type resolution hierarchy; "
                                "DTypes of two matching loops do not have "
                                "a clear hierarchy defined. Diamond shape "
                                "inheritance is unsupported for use with "
                                "UFunc type resolution. (You may resolve "
                                "this by inserting an additional common "
                                "subclass). This limitation may be "
                                "partially resolved in the future.");
                        return -1;
                    }
                    if (new_is_subclass) {
                        unambiguous_equivally_good = 0;
                        best = 1;
                    }
                    else {
                        unambiguous_equivally_good = 0;
                        best = 2;
                    }
                }
                if ((current_best != -1) && (current_best != best)) {
                    /*
                     * We need a clear best, this could be tricky, unless
                     * the signature is identical, we would have to compare
                     * against both of the found ones until we find a
                     * better one.
                     * Instead, only support the case where they are
                     * identical.
                     */
                    // TODO: Document the above comment (figure out if OK)
                    current_best = -1;
                    break;
                }
                current_best = best;
            }

#if NPY_UF_DBG_TRACING
            printf("Comparison between the two tuples gave %d\n ",
                   current_best);
            PyObject_Print(best_dtypes, stdout, 0);
            PyObject_Print(curr_dtypes, stdout, 0);
            printf(" in ufunc %s\n", ufunc->name);
#endif
            if (current_best == -1) {
                if (unambiguous_equivally_good) {
                    /* unset the best resolver to indicate this */
                    best_resolver_info = NULL;
                    continue;
                }
                PyErr_SetString(PyExc_TypeError,
                        "Could not resolve UFunc loop, two loops "
                        "matched equally well.");
                return -1;
            }
            else if (current_best == 0) {
                /* The new match is not better, continue looking. */
                continue;
            }
        }
        /* The new match is better (or there was no previous match) */
        best_dtypes = curr_dtypes;
        best_resolver_info = resolver_info;
    }
    if (best_dtypes == NULL) {
        /* The non-legacy lookup failed */
        *out_info = NULL;
        return 0;
    }

    if (best_resolver_info == NULL) {
        /*
         * This happens if two were equal, but we kept searching
         * for a better one.
         */
        PyErr_SetString(PyExc_TypeError,
                "Could not resolve UFunc loop, two loops "
                "matched equally well.");
        return -1;
    }

    *out_info = best_resolver_info;
    return 0;
}


/*
 * Used for the legacy fallback promotion when `signature` or `dtype` is
 * provided.
 * We do not need to pass the type tuple when we use the legacy path
 * for type resolution rather than promotion; the old system did not
 * differentiate between these two concepts.
 */
static int
_make_new_typetup(
        int nop, PyArray_DTypeMeta *signature[], PyObject **out_typetup) {
    *out_typetup = PyTuple_New(nop);
    if (*out_typetup == NULL) {
        return -1;
    }

    int none_count = 0;
    for (int i = 0; i < nop; i++) {
        PyObject *item;
        if (signature[i] == NULL) {
            item = Py_None;
            none_count++;
        }
        else {
            if (!signature[i]->legacy || signature[i]->abstract) {
                /*
                 * The legacy type resolution can't deal with these.
                 * This path will return `None` or so in the future to
                 * set an error later if the legacy type resolution is used.
                 */
                PyErr_SetString(PyExc_RuntimeError,
                        "Internal NumPy error: new DType in signature not yet "
                        "supported. (This should be unreachable code!)");
                Py_SETREF(*out_typetup, NULL);
                return -1;
            }
            item = (PyObject *)signature[i]->singleton;
        }
        Py_INCREF(item);
        PyTuple_SET_ITEM(*out_typetup, i, item);
    }
    if (none_count == nop) {
        /* The whole signature was None, simply ignore type tuple */
        Py_DECREF(*out_typetup);
        *out_typetup = NULL;
    }
    return 0;
}


/*
 * Legacy type resolution unfortunately works on the original array objects
 * and we have no choice but to pass them in.
 */
static int
legacy_resolve_implementation_info(PyUFuncObject *ufunc,
        PyArrayObject *const *ops, PyArray_DTypeMeta *signature[],
        PyObject **out_info)
{
    int nargs = ufunc->nargs;
    PyArray_Descr *out_descrs[NPY_MAXARGS] = {NULL};
    PyObject *type_tuple = NULL;
    if (_make_new_typetup(nargs, signature, &type_tuple) < 0) {
        return NULL;
    }

    /*
     * We use unsafe casting. This is of course not accurate, but that is OK
     * here, because for promotion/dispatching the casting safety makes no
     * difference.  Whether the actual operands can be casts must be checked
     * during the type resolution step (which may _also_ calls this!).
     */
    if (ufunc->type_resolver(ufunc,
            NPY_UNSAFE_CASTING, (PyArrayObject **)ops, type_tuple,
            out_descrs) < 0) {
        goto error;
    }
    PyObject *DType_tuple = PyTuple_New(nargs);
    if (DType_tuple == NULL) {
        goto error;
    }
    Py_XDECREF(type_tuple);
    for (int i = 0; i < nargs; i++) {
        PyObject *DType = (PyObject *)NPY_DTYPE(out_descrs[i]);
        Py_INCREF(DType);
        PyTuple_SET_ITEM(DType_tuple, i, DType);
    }
    /* no need for goto error anymore */

    PyArray_NewLegacyWrappingArrayMethod()

    return 0;

  error:
    Py_XDECREF(type_tuple);
    for (int i = 0; i < nargs; i++) {
        Py_XDECREF(out_descrs[i]);
    }
    return -1;
}


/*
 * The central entrypoint for the promotion and dispatching machinery.
 * It currently works with the operands (although it would be possible to
 * only work with DType (classes/types).
 */
NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[], PyArray_DTypeMeta *signature[])
{
    int nargs = ufunc->nargs;

    /*
     * Get the actual DTypes we operate with by mixing the operand array
     * ones with the passed signature.
     */
    PyArray_DTypeMeta *op_dtypes[NPY_MAXARGS];
    for (int i = 0; i < nargs; i++) {
        if (signature[i] != NULL) {
            /*
             * ignore the input here, we cannot overwrite signature yet
             * since it is fixed (cannot be promoted!)
             */
            op_dtypes[i] = signature[i];
        }
        if (ops[i] == NULL) {
            op_dtypes[i] = NULL;
        }
        else {
            /*
             * TODO: This path will have to check for when an object was
             * "originally" a Python scalar. (Probably flagging it on the
             * object.)  Alternatively, this needs to be done earlier.
             */
            op_dtypes[i] = NPY_DTYPE(PyArray_DTYPE(ops[i]));
        }
    }

    /*
     * Fetch the dispatching info which consists of the implementation and
     * the DType signature tuple.  There are three steps:
     *
     * 1. Check the cache.
     * 2. Check all registered loops/promoters to find the best match.
     * 3. Fall back to the legacy implementation if no match was found.
     */
    PyObject *info = PyArrayIdentityHash_GetItem(
            (PyArrayIdentityHash *)ufunc->_dispatch_cache, (PyObject **)op_dtypes);

    if (NPY_UNLIKELY(info == NULL)) {
        if (resolve_implementation_info(ufunc, op_dtypes, &info) < 0) {
            return NULL;
        }
        if NPY_UNLIKELY(info == NULL) {
            /*
             * One last try by using the legacy type resolver (this may
             * succeed even if the final resolution is invalid because there
             * is no matching loop.
             */
            if (legacy_resolve_implementation_info(ufunc,
                    ops, signature, &info) < 0) {
                return NULL;
            }
        }
    }
    assert(PyTuple_CheckExact(info) && PyTuple_GET_SIZE(info) == 2);

    /* Make an exact check to make abuse hard for now */
    if (Py_TYPE(PyTuple_GET_ITEM(info, 1)) != &PyArrayMethod_Type) {
        PyErr_SetString(PyExc_NotImplementedError,
                "Promoters are not implemented yet, they will be called here "
                "and then store the result (or the promoter itself)");
        /*
         * int res = call_ufuncimpl_resolver(
         *        best_resolver, ufunc, signature, ufunc_impl);
         * if (res < 0) {
         *     return -1;
         * }
         * if (!is_any_dtype_abstract) {
         *     // No abstract dtype, so store ufunc_impl and not function
         *     best_resolver = (PyObject *)*ufunc_impl;
         * }
         */
    }


}