#ifndef _NPY_DISPATCHING_H
#define _NPY_DISPATCHING_H

#include "numpy/ufuncobject.h"
#include "array_method.h"


NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject * const ops[], PyArray_DTypeMeta *signature[]);



#endif  /*_NPY_DISPATCHING_H */
