/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"
#include "npy_ctypes.h"
#include "npy_pycompat.h"

#include "_datetime.h"
#include "common.h"
#include "alloc.h"
#include "assert.h"

#include "dtypemeta.h"


static PyMemberDef dtypemeta_members[] = {
    {"abstract",
        T_INT, offsetof(PyArray_DTypeMeta, abstract), READONLY, NULL},
    {"type",
        T_OBJECT, offsetof(PyArray_DTypeMeta, typeobj), READONLY, NULL},
    {"kind",
        T_CHAR, offsetof(PyArray_DTypeMeta, kind), READONLY, NULL},
    {"char",
        T_CHAR, offsetof(PyArray_DTypeMeta, type), READONLY, NULL},
    {"num",
        T_INT, offsetof(PyArray_DTypeMeta, type_num), READONLY, NULL},
    {"itemsize",
        T_INT, offsetof(PyArray_DTypeMeta, elsize), READONLY, NULL},
    {"flags",
        T_BYTE, offsetof(PyArray_DTypeMeta, flags), READONLY, NULL},
    {"flexible",
        T_INT, offsetof(PyArray_DTypeMeta, flexible), READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static void
dtypemeta_dealloc(PyArray_DTypeMeta *self)
{
    printf("INSIDE DEALLOC!!!!!!\n");
    (&PyType_Type)->tp_dealloc((PyObject *)self);
}

static PyObject *
dtypemeta_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *self = (&PyType_Type)->tp_new(type, args, kwds);
    printf("INSIDE NEW: created at %ld from type %ld and type is %ld \n", (size_t)self, (size_t)type, (size_t)&PyType_Type);
    return NULL;
}

static int
dtypemeta_init(PyObject *type, PyObject *args, PyObject *kwds)
{
    int res = (&PyType_Type)->tp_init(type, args, kwds);
    printf("INSIDE INIT\n");
    return res;
}


//static int
//dtypemeta_repr(PyObject *type, PyObject *args, PyObject *kwds)
//{
//    printf("INSIDE REPR\n");
//    return Py_BuildValue("s", "<this_is_a_funny_repr>");
//}
//
//
//static int
//dtypemeta_str(PyObject *type, PyObject *args, PyObject *kwds)
//{
//    printf("INSIDE STR\n");
//    return Py_BuildValue("s", "<this_is_a_funny_str>");
//}
//
//PyObject *
//dtypeclass_rawnew(char *name) {
//    // TODOL Should be NN, but I am desperate to find the bug...
//    PyObject *args = Py_BuildValue("(sOO)", name, PyTuple_Pack(1, &PyArrayDescr_Type), PyDict_New());
//
//    Py_INCREF(&PyArrayDTypeMeta_Type);
//    PyObject *res = dtypemeta_new(&PyArrayDTypeMeta_Type, args, NULL);
//    dtypemeta_init(res, args, NULL);
//
//    Py_DECREF(args);
//    return res;
//}


static int
dtypemeta_traverse(PyTypeObject *type, visitproc visit, void *arg)
{
    printf("eehh wah! Traversing (doesn't happen, right?)!\n");
    return 0;
}

static int
dtypemeta_clear(PyTypeObject *type)
{
    printf("eehh wah! Clearing (doesn't happen, right?)!\n");
    return 0;
}


/*
 * This is brutal. Because it seems tricky to do otherwise, use
 * the static full Python API on malloc allocated objects, so that they
 * are allocated on the heap, but specifically not heaptype objects.
 *
 * This is the version wrapping an old-style dtype.
 */
NPY_NO_EXPORT int
descr_dtypesubclass_init(PyArray_Descr *dtype) {
    PyTypeObject *dtype_classobj = calloc(1, sizeof(PyArray_DTypeMeta));
    if (dtype_classobj == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    dtypemeta_slots *dt_slots = calloc(1, sizeof(dtypemeta_slots));
    if (dt_slots == NULL) {
        free(dtype_classobj);
        PyErr_NoMemory();
        return -1;
    }

    memcpy(dtype_classobj, &PyArrayDescr_Type, sizeof(PyArrayDescr_Type));
    Py_TYPE(dtype_classobj) = &PyArrayDTypeMeta_Type;
    dtype_classobj->tp_base = &PyArrayDescr_Type;
    PyType_Ready(dtype_classobj);

    Py_INCREF(dtype_classobj);  // manually allocated, shouldn't matter?
    Py_TYPE(dtype) = dtype_classobj;
    // printf("Attached class to %ld\n", dtype);
    
    /* Shiboken2 does this, so assume it may be necessary???? */
    //dtype->tp_weaklistoffset = offsetof(PyArray_Descr, weakreflist);
    //dtype->tp_dictoffset = offsetof(PyArray_Descr, ob_dict);

    PyArray_DTypeMeta *dtype_class = (PyArray_DTypeMeta *)dtype_classobj;
    dtype_class->typeobj = dtype->typeobj;
    dtype_class->kind = dtype->kind;
    dtype_class->type = dtype->type;
    dtype_class->flags = dtype->flags;
    dtype_class->elsize = dtype_class->elsize;
    dtype_class->flexible = dtype->elsize == 0;  // Silly but true here...
    dtype_class->abstract = 0;
    dtype_class->type_num = dtype->type_num;
    dtype_class->f = NULL;  // dtype->f;
    // Just hold on to a reference to name:
    ((PyTypeObject *)dtype_class)->tp_name = dtype->typeobj->tp_name;

    // This seems like it might make sense (but probably not here):
    Py_INCREF(dtype);
    //PyDict_SetItemString(dict, "sometype", dtype_classobj);
}


NPY_NO_EXPORT PyTypeObject PyArrayDTypeMeta_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "numpy.DTypeMeta",                          /* tp_name */
    sizeof(PyArray_DTypeMeta),              /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)dtypemeta_dealloc,            /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    (void *)0,                                  /* tp_reserved */
    0,                  /* tp_repr */
    0,                           /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                   /* tp_str */
    0,                    /* tp_getattro */
    0,                    /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "nonsense docs",                                          /* tp_doc */
    dtypemeta_traverse,                                          /* tp_traverse */
    dtypemeta_clear,                                          /* tp_clear */
    0,        /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                         /* tp_methods */
    dtypemeta_members,                         /* tp_members */
    0,                         /* tp_getset */
    &PyType_Type,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)dtypemeta_init,                            /* tp_init */
    0,                                          /* tp_alloc */
    dtypemeta_new,                             /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
