#ifndef _NPY_CASTINGIMPL_H_
#define _NPY_CASTINGIMPL_H_

#include "dtypemeta.h"


NPY_NO_EXPORT PyObject *
castingimpl_legacynew(
        PyArray_DTypeMeta *from_dtype,
        PyArray_DTypeMeta *to_dtype);

struct _CastingImpl;

typedef int (adjust_descriptors_func)(
        struct _CastingImpl *self,
        PyArray_Descr *in_descrs[2], PyArray_Descr *out_descrs[2],
        NPY_CASTING NPY_UNUSED(casting));

/*
 * This struct requires a rehaul, it must be compatible with the UFuncImpl
 * one. And should be extensible (and user subclassable) in the future.
 * This means that the actual public API should be short and possibly shared
 * with UFuncImpl.
 * Even if it is short enough to fully define it, we should add at least
 * one reserved slot which can point to a dynamically allocated (and opaque)
 * layer similar to the DTypeMeta field.
 */
typedef struct _CastingImpl {
    PyObject_HEAD
    PyArray_DTypeMeta *from_dtype;
    PyArray_DTypeMeta *to_dtype;
    adjust_descriptors_func *adjust_descriptors;
    void *get_transferfunction;
} CastingImpl;


#endif  /* _NPY_CASTINGIMPL_H_ */
