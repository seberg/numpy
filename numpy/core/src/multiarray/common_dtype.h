#ifndef _NPY_COMMON_DTYPE_H_
#define _NPY_COMMON_DTYPE_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <numpy/ndarraytypes.h>
#include "dtypemeta.h"

NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2);


#endif  /* _NPY_COMMON_DTYPE_H_ */
