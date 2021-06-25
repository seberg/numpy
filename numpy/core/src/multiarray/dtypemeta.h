#ifndef _NPY_DTYPEMETA_H
#define _NPY_DTYPEMETA_H

#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
/*
 * This function will hopefully be phased out or replaced, but was convenient
 * for incremental implementation of new DTypes based on DTypeMeta.
 * (Error checking is not required for DescrFromType, assuming that the
 * type is valid.)
 */
static NPY_INLINE PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}


/*
 * Returns a new reference to type if it is already NBO, otherwise
 * returns a copy converted to NBO.
 * For non-builtin types (and void) this is not necessarily byte-order only,
 * but a general "canonical" property.
 */
static NPY_INLINE PyArray_Descr *
ensure_dtype_canonical(PyArray_Descr *type)
{
    /*
     * It would be nice to add a flag to the DType, instead of always
     * calling the actual function.
     */
    return NPY_DTYPE(type)->ensure_canonical(type);
}


NPY_NO_EXPORT int
dtypemeta_wrap_legacy_descriptor(PyArray_Descr *dtypem);

#endif  /*_NPY_DTYPEMETA_H */
