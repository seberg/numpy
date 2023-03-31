#!/usr/bin/env python3
import os
import argparse

import genapi
from genapi import \
        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

import numpy_api

# use annotated api when running under cpychecker
h_template = r"""
#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;
extern NPY_NO_EXPORT PyTypeObject PyArrayNeighborhoodIter_Type;
extern NPY_NO_EXPORT PyBoolScalarObject _PyArrayScalar_BoolValues[2];

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API=NULL;
#endif
#endif

%s


/* NPY_MAXDIMS is actually version depended (wouldn't be defined here) */
#undef NPY_MAXDIMS
#define NPY_MAXDIMS 64
#define NPY_MAXDIMS_ACTUAL (*(int *)PyArray_API[308])


#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)

#if NPY_TARGET_VERSION < 0x02000000
/*
 * Backcompat version of a new API call, this could also be
 * an inline function that has an `if` based on table content.
 * In this case, the first version does a "stupid" assignment
 * while the second one can do a proper cast (but this can be
 * tweaked).
 * Using `PyArray_PackIntoArr` as abstraction because it works
 * clearer on older NumPy versions.
 */
static inline int  // inline, because its OK if unused...
_NPY_PyArray_PackIntoArr(PyArrayObject *arr, char *ptr, PyObject *obj)
{
    return PyArray_DESCR(arr)->f->setitem(obj, ptr, arr);
}
#else
#define PyArray_PackIntoArr(arr, ptr, object) \
    PyArray_Pack(PyArray_DESCR(arr), ptr, obj)
#endif

/*
 * This code translated the existing NumPy 1.x table to a NumPy 2
 * one.
 * It would be in a dedicate header and such a step is only necessary
 * for a major release.  Ideal, would be major that there are never more
 * than 2 major releases "backported" like this, which gives us a major
 * release about every 3.5 years.
 */
static void *
_npy_make_numpy2_table(void **c_api) {
    printf("Creating new NumPy 2 backcompat table!\n");
    // TODO: Basically, 304 is currently right, lets add some more:
    // NOTE: Also add +50, so we have space in case we realize that
    //       we need more symbols and want to add them in a 2.1!
    //       And we make sure that this space is NULLed (Calloc).
    void **numpy2_table = (void **)PyObject_Calloc((310+50), sizeof(void *));
    if (numpy2_table == NULL) {
        return NULL;
    }
    // For now, change nothing (we need to change our numbering, since
    // we always use the new table/numbers)!
    memcpy(numpy2_table, c_api, 307*sizeof(void *));
    static int maxdims = 32;

    numpy2_table[307] = &maxdims;  // MAXDIMS, make pretty!
    numpy2_table[308] = (void *)&_NPY_PyArray_PackIntoArr;

    return numpy2_table;
}


static int
_import_array(void)
{
  int st;
  int using_numpy_2 = 0;
  PyObject *numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  PyObject *c_api = NULL;

  if (numpy == NULL) {
      return -1;
  }
  if (PyObject_HasAttrString(numpy, "_ARRAY_API2")) {
    /* Running on NumPy 2, so can use the API table normally */
    c_api = PyObject_GetAttrString(numpy, "_ARRAY_API2");
    using_numpy_2 = 1;
  }
  else {
#if NPY_TARGET_VERSION >= 0x02000000
    /* compiled without support for NumPy 2.x */
    Py_DECREF(numpy);
    PyErr_Format(PyExc_RuntimeError,
        "Extension module was compiled solely for use with NumPy 2 "
        "and higher.  Please XYZ about compiling against NumPy.");
    return -1;
#else
    /* Assume we are on NumPy 1.x */
    c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
#endif
  }
  Py_DECREF(numpy);
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_ARRAY_API not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  void **api_table = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (api_table == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
      return -1;
  }

  if (!using_numpy_2) {
    /* 
     * Translate NumPy 1 table (if not done yet).  Stuff it into the
     * first slot (which is always a pointer to NULL).
     * Could also make a new Capsule and attach...
     */
     printf("NumPy 2 table already created? %%p != 0\n", *(void **)api_table[1]);
    if (*(void **)api_table[1] != NULL) {
        /* We already created and tagged on a new C-API version */
        api_table = (void **)api_table[1];
    }
    else {
        api_table[1] = (void *)_npy_make_numpy2_table(api_table);
        if (api_table[1] == NULL) {
            return -1;
        }
        api_table = (void **)api_table[1];
        printf("NumPy 2 table created? %%p != 0\n", api_table);
    }
  }

  PyArray_API = api_table;

  /* Perform runtime check of C API version */
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "ABI version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
      return -1;
  }
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "API version 0x%%x but this version of numpy is 0x%%x . "\
             "Check the section C-API incompatibility at the "\
             "Troubleshooting ImportError section at "\
             "https://numpy.org/devdocs/user/troubleshooting-importerror.html"\
             "#c-api-incompatibility "\
              "for indications on how to solve this problem .", \
             (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
      return -1;
  }

  /*
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as unknown endian");
      return -1;
  }
#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as big endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  if (st != NPY_CPU_LITTLE) {
      PyErr_SetString(PyExc_RuntimeError,
                      "FATAL: module compiled as little endian, but "
                      "detected different endianness at runtime");
      return -1;
  }
#endif

  return 0;
}

#define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return NULL; } }

#define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }

#define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }

#endif

#endif
"""


c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
%s
};
"""

def generate_api(output_dir, force=False):
    basename = 'multiarray_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    targets = (h_file, c_file)

    sources = numpy_api.multiarray_api

    if (not force and not genapi.should_rebuild(targets, [numpy_api.__file__, __file__])):
        return targets
    else:
        do_generate_api(targets, sources)

    return targets

def do_generate_api(targets, sources):
    header_file = targets[0]
    c_file = targets[1]

    global_vars = sources[0]
    scalar_bool_values = sources[1]
    types_api = sources[2]
    multiarray_funcs = sources[3]

    multiarray_api = sources[:]

    module_list = []
    extension_list = []
    init_list = []

    # Check multiarray api indexes
    multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
    genapi.check_api_dict(multiarray_api_index)

    numpyapi_list = genapi.get_api_functions('NUMPY_API',
                                             multiarray_funcs)

    # Create dict name -> *Api instance
    api_name = 'PyArray_API'
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name][0]
        annotations = multiarray_funcs[name][1:]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations,
                                                  f.return_type,
                                                  f.args, api_name, version_limit=f.version_limit)

    for name, val in global_vars.items():
        index, type = val
        multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name)

    for name, val in scalar_bool_values.items():
        index = val[0]
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name)

    for name, val in types_api.items():
        index = val[0]
        internal_type =  None if len(val) == 1 else val[1]
        multiarray_api_dict[name] = TypeApi(
            name, index, 'PyTypeObject', api_name, internal_type)

    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        raise AssertionError(
            "Multiarray API size mismatch - "
            "index has extra keys {}, dict has extra keys {}"
            .format(keys_index - keys_dict, keys_dict - keys_index)
        )

    extension_list = []
    for name, index in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        extension_list.append(api_item.define_from_array_api_string())
        init_list.append(api_item.array_api_define())
        module_list.append(api_item.internal_define())

    # Write to header
    s = h_template % ('\n'.join(module_list), '\n'.join(extension_list))
    genapi.write_file(header_file, s)

    # Write to c-code
    s = c_template % ',\n'.join(init_list)
    genapi.write_file(c_file, s)

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Path to the output directory"
    )
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="An ignored input - may be useful to add a "
             "dependency between custom targets"
    )
    args = parser.parse_args()

    outdir_abs = os.path.join(os.getcwd(), args.outdir)

    generate_api(outdir_abs)


if __name__ == "__main__":
    main()
