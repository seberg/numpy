#ifndef _NPY_DISPATCHING_H
#define _NPY_DISPATCHING_H

#define _UMATHMODULE

#include <numpy/ufuncobject.h>
#include "array_method.h"


NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[], int force_legacy_promotion);


#endif  /*_NPY_DISPATCHING_H */
