#ifndef _NPY_LEGACY_ARRAY_METHOD_H
#define _NPY_LEGACY_ARRAY_METHOD_H

#include "numpy/ndarraytypes.h"
#include "array_method.h"


NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *,
        PyArray_DTypeMeta **, PyArray_Descr **, PyArray_Descr **);

#endif  /*_NPY_LEGACY_ARRAY_METHOD_H */
