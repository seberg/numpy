#ifndef _NPY_NPY_HASHTABLE_H
#define _NPY_NPY_HASHTABLE_H

#include "numpy/ndarraytypes.h"


typedef struct {
    int key_len;  /* number of identities used */
    /* Buckets stores: val1, key1[0], key1[1], ..., val2, key2[0], ... */
    PyObject **buckets;
    npy_intp size;  /* current size */
    npy_intp nelem;  /* number of elements */
} PyArrayIdentityHash;


NPY_NO_EXPORT int
PyArrayIdentityHash_SetItem(
        PyArrayIdentityHash *tb, PyObject *const *key, PyObject *value);

NPY_NO_EXPORT PyObject *
PyArrayIdentityHash_GetItem(PyArrayIdentityHash const *tb, PyObject *const *key);

NPY_NO_EXPORT PyArrayIdentityHash *
PyArrayIdentityHash_New(int key_len);

void
PyArrayIdentityHash_Dealloc(PyArrayIdentityHash *tb);

#endif  /* _NPY_NPY_HASHTABLE_H */
