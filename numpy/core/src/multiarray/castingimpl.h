#ifndef _NPY_CASTINGIMPL_H_
#define _NPY_CASTINGIMPL_H_

#include "dtypemeta.h"
#include "lowlevel_strided_loops.h"


NPY_NO_EXPORT PyObject *
castingimpl_legacynew(
        PyArray_DTypeMeta *from_dtype,
        PyArray_DTypeMeta *to_dtype);

//struct _CastingImpl;

/*
 * Adjust and check for casting. Right now this function assumes that a rough
 * check is already done. For example casting float64 -> float32 is unsafe,
 * so if we are doing safe casting, we should not end up here at all.
 * (the trivial implementation could be inlined/optimized as NULL)
 */
typedef int (adjust_descriptors_func)(
            struct _CastingImpl *self,
            PyArray_Descr *in_descrs[2], PyArray_Descr *out_descrs[2],
            NPY_CASTING casting);

typedef int (get_transferfunction_func)(
            struct _CastingImpl *self,
            int aligned,
            npy_intp src_stride, npy_intp dst_stride,
            PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
            int move_references,
            PyArray_StridedUnaryOp **out_stransfer,
            NpyAuxData **out_transferdata,
            int *out_needs_api);

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
    get_transferfunction_func *get_transferfunction;
} CastingImpl;


#endif  /* _NPY_CASTINGIMPL_H_ */
