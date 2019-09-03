#ifndef _NPY_ABSTRACTDTYPE_H_
#define _NPY_ABSTRACTDTYPE_H_

#include "dtypemeta.h"



typedef struct {
    PyArray_DTypeMeta super;
    PyObject *minimum;
    PyObject *maximum;
} PyArray_PyValueAbstractDType;

NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyIntAbstractDType;
NPY_NO_EXPORT PyArray_PyValueAbstractDType PyArray_PyFloatAbstractDType;

#endif  /* _NPY_ABSTRACTDTYPE_H_ */
