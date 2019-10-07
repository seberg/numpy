/* Array Descr Object */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarraytypes.h>
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
#include "castingimpl.h"
#include "convert_datatype.h"
#include "abstractdtype.h"


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
    {"flexible",
        T_INT, offsetof(PyArray_DTypeMeta, flexible), READONLY, NULL},
    {NULL, 0, 0, 0, NULL},
};

static void
dtypemeta_dealloc(PyArray_DTypeMeta *self) {
    Py_DECREF(self->typeobj);
    free(self->dt_slots);
    // TODO: I guess types that can run into this should just be heaptypes...?
    if (self->super.ht_type.tp_flags & Py_TPFLAGS_HEAPTYPE) {
        (&PyType_Type)->tp_dealloc((PyObject *) self);
    }
    else {
        PyObject_Del(self);
    }
}

static PyObject *
dtypemeta_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    // PyObject *self = (&PyType_Type)->tp_new(type, args, kwds);

    PyErr_SetString(PyExc_TypeError,
        "Cannot subclass dtype (abstract dtypes will be subclassable)");
    return NULL;
}

static int
dtypemeta_init(PyObject *type, PyObject *args, PyObject *kwds)
{
    int res = (&PyType_Type)->tp_init(type, args, kwds);
    if (res < 0) {
        return res;
    }
    PyErr_SetString(PyExc_TypeError,
        "init slot of dtypemeta should currently never be called!");
    return -1;
}


static CastingImpl *
legacy_can_cast(
        PyArray_DTypeMeta *from_dtype,
        PyArray_DTypeMeta *to_dtype,
        NPY_CASTING casting)
{
    // TODO: Practically doesn't matter, but error checks missing.
    PyArray_Descr *from_descr = PyArray_DescrNewFromType(from_dtype->type_num);
    PyArray_Descr *to_descr = PyArray_DescrNewFromType(to_dtype->type_num);
    if (!PyArray_LegacyCanCastTypeTo(from_descr, to_descr, casting)) {
        Py_DECREF(from_descr);
        Py_DECREF(to_descr);
        Py_INCREF(Py_NotImplemented);
        return (CastingImpl *)Py_NotImplemented;
    }
    Py_DECREF(from_descr);
    Py_DECREF(to_descr);
    /* should not return a new one every time of course: */
    return (CastingImpl *)castingimpl_legacynew(from_dtype, to_dtype);
}


static CastingImpl *
legacy_can_cast_to(
        PyArray_DTypeMeta *from_dtype,
        PyArray_DTypeMeta *to_dtype,
        NPY_CASTING casting)
{
    if (!to_dtype->is_legacy_wrapper) {
        Py_INCREF(Py_NotImplemented);
        return (CastingImpl *)Py_NotImplemented;
    }
    if (from_dtype->type_num >= to_dtype->type_num) {
        // Reject to make things a bit more interesting. Also makes a
        // check which only wants to know if casting is possible faster.
        Py_INCREF(Py_NotImplemented);
        return (CastingImpl *)Py_NotImplemented;
    }
    return legacy_can_cast(from_dtype, to_dtype, casting);
}


static CastingImpl *
legacy_can_cast_from(
        PyArray_DTypeMeta *to_dtype,
        PyArray_DTypeMeta *from_dtype,
        NPY_CASTING casting)
{
    if (!from_dtype->is_legacy_wrapper) {
        Py_INCREF(Py_NotImplemented);
        return (CastingImpl *)Py_NotImplemented;
    }
    if (to_dtype->type_num >= from_dtype->type_num) {
        // Reject to make things a bit more interesting.
        Py_INCREF(Py_NotImplemented);
        return (CastingImpl *)Py_NotImplemented;
    }
    return legacy_can_cast(from_dtype, to_dtype, casting);
}


static PyArray_Descr *
legacy_default_descr(PyArray_DTypeMeta *cls) {
    // For current flexible types (strings and void, but also
    // datetime/timedelta this practically exists currently, but should
    // not exist.
    return PyArray_DescrFromType(cls->type_num);
}

static PyArray_Descr *
discover_descr_using_default(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    return cls->dt_slots->default_descr(cls);
}


static PyArray_DTypeMeta*
legacy_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    assert(cls != other);
    assert(cls->type_num != other->type_num);
    if (!other->is_legacy_wrapper) {
        Py_INCREF(Py_NotImplemented);
        // TODO: Maybe should fix the function type instead?
        return (PyArray_DTypeMeta*)Py_NotImplemented;
    }

    if (cls->type_num >= other->type_num) {
        // Reject to make things a bit more interesting.
        Py_INCREF(Py_NotImplemented);
        return (PyArray_DTypeMeta*)Py_NotImplemented;
    }

    PyArray_Descr *descr1 = PyArray_DescrFromType(cls->type_num);
    PyArray_Descr *descr2 = PyArray_DescrFromType(other->type_num);

    PyArray_Descr *common_descr = PyArray_LegacyPromoteTypes(descr1, descr2);
    Py_DECREF(descr1);
    Py_DECREF(descr2);
    if (common_descr == NULL) {
        return NULL;
    }
    PyArray_DTypeMeta *common = (PyArray_DTypeMeta *)Py_TYPE(common_descr);
    Py_INCREF(common);
    Py_DECREF(common_descr);
    return common;
}

static PyArray_DTypeMeta*
object_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /* Object dtype always wins boojah! */
    Py_INCREF(cls);
    return cls;
}

static PyArray_Descr*
legacy_common_instance(
        PyArray_DTypeMeta *cls, PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    return PyArray_LegacyPromoteTypes(descr1, descr2);
}

static PyArray_Descr*
VOID_common_instance(
        PyArray_DTypeMeta *cls, PyArray_Descr *descr1, PyArray_Descr *descr2) {
    PyArray_Descr *out_descr = NULL;
    /* The legacy code thinks that V8 and V9 cannot promote; lets disagree. */
    if ((descr1->names == NULL) && (descr1->subarray == NULL) &&
        (descr1->names == NULL) && (descr1->subarray == NULL)) {
        if (descr1->elsize >= descr2->elsize) {
            out_descr = descr1;
        }
        else {
            out_descr = descr2;
        }
    }
    /*
     * If they have fields and subarray, they cannot be promoted, but
     * they could be equivalent, in which case all is OK.
     * (Code taken from LegacyPromoteTypes, may be possible to simplify)
     */
    if (PyArray_CanCastTypeTo(descr2, descr1, NPY_EQUIV_CASTING)) {
        out_descr = descr1;
    }
    if (out_descr != NULL) {
        if (PyArray_ISNBO(descr1->byteorder)) {
            Py_INCREF(descr1);
            return descr1;
        }
        else {
            return PyArray_DescrNewByteorder(descr1, NPY_NATIVE);
        }
    }
    PyErr_SetString(PyExc_TypeError,
            "cannot find common VOID dtype instance unless they are "
            "equivalent or do not have fields/subarrays.");
    return NULL;
}

static PyArray_Descr *
string_discover_descr_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj)
{
    /* We disallow sequences during settings, so do it here as well */
    // TODO: If we have a ragged array, we could never go here, since strings
    //       will not support this anyway. This way, such an array could
    //       create a huge intermediate before actually erroring out, but OK.
    Py_ssize_t length;
    if (PyBytes_Check(obj)) {
        length = PyString_GET_SIZE(obj);
        if (cls->type_num == NPY_UNICODE) {
            length *= 4;
        }
    }
    else {
        PyObject * str = PyObject_Str(obj);
        if (str == NULL) {
            // TODO: Probably might as well just raise the error here?
            PyErr_Clear();
            return cls->dt_slots->default_descr(cls);
        }
        if (cls->type_num != NPY_UNICODE) {
            /* This can also be a VOID type... */
            length = PyUnicode_GetLength(str);
        } else {
            length = PyUnicode_GET_DATA_SIZE(str);
#ifndef Py_UNICODE_WIDE
            length <<= 1;
#endif
        }
        Py_DECREF(str);
        if (length > NPY_MAX_INT) {
            PyErr_SetString(PyExc_TypeError,
                            "string representation of object is too large to store "
                            "as NumPy string fixed width string.");
            return NULL;
        }
    }
    if (length == 0) {
        // TODO: Legacy behaviour of not allowing empty strings!
        length = cls->type_num == NPY_UNICODE ? 4 : 1;
    }
    PyArray_Descr *descr = PyArray_DescrNewFromType(cls->type_num);
    if (descr == NULL) {
        return NULL;
    }
    descr->elsize = (int)length;
    return descr;
}

static PyArray_Descr *
discover_datetime_and_timedelta_from_pyobject(
        PyArray_DTypeMeta *cls, PyObject *obj) {
    if (PyArray_IsScalar(obj, Datetime) ||
                PyArray_IsScalar(obj, Timedelta)) {
        PyArray_DatetimeMetaData *meta;
        PyArray_Descr *descr = PyArray_DescrFromScalar(obj);
        meta = get_datetime_metadata_from_dtype(descr);
        if (meta == NULL) {
            return NULL;
        }
        PyArray_Descr *new_descr = create_datetime_dtype(cls->type_num, meta);
        Py_DECREF(descr);
        return new_descr;
    }
    else {
        return find_object_datetime_type(obj, cls->type_num);
    }
}


static PyArray_Descr *
legacy_ensure_native(PyArray_Descr *descr) {
    if (PyDataType_ISNOTSWAPPED(descr)) {
        Py_INCREF(descr);
        return descr;
    }
    return PyArray_DescrNewByteorder(descr, NPY_NATIVE);
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
    // TODO: These are not exactly heaptypes (yet?)...
    //       would need PyObject_new otherwise?
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

    // TODO: Does this break a few fields in principle?
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
    dtype_class->is_legacy_wrapper = NPY_TRUE;

    Py_INCREF(dtype->typeobj);
    dtype_class->typeobj = dtype->typeobj;
    dtype_class->kind = dtype->kind;
    dtype_class->type = dtype->type;
    dtype_class->flags = dtype->flags;
    dtype_class->elsize = dtype_class->elsize;
    if (PyTypeNum_ISDATETIME(dtype->type_num) ||
                PyTypeNum_ISFLEXIBLE(dtype->type_num)) {
        /* Datetimes are flexible in this sense due to the attached unit. */
        dtype_class->flexible = 1;
    }
    dtype_class->abstract = 0;
    dtype_class->type_num = dtype->type_num;
    dtype_class->f = NULL;  // dtype->f;
    // Just hold on to a reference to name:
    ((PyTypeObject *)dtype_class)->tp_name = dtype->typeobj->tp_name;

    // Register the typeobject for quicker discovery globally.
    int success = PyDict_SetItem(
            PyArrayDTypeMeta_associated_types,
            (PyObject *)dtype_class->typeobj, (PyObject *)dtype_class);
    if (success < 0) {
        // TODO: Need to clean up in this unlikely event.
        return -1;
    }
    if (dtype_class->type_num == NPY_BOOL) {
        success = PyDict_SetItem(
                PyArrayDTypeMeta_associated_types,
                (PyObject *)&PyBool_Type, (PyObject *)dtype_class);
        if (success < 0) {
            // TODO: Need to clean up in this unlikely event.
            return -1;
        }
    }
    // TODO: Also register the (super common) double, but will need to be abstract.
    if (dtype_class->type_num == NPY_DOUBLE) {
        success = PyDict_SetItem(
                PyArrayDTypeMeta_associated_types,
                (PyObject *)&PyFloat_Type, (PyObject *)dtype_class);
        if (success < 0) {
            // TODO: Need to clean up in this unlikely event.
            return -1;
        }
    }
    if (dtype_class->type_num == NPY_STRING) {
        success = PyDict_SetItem(
                PyArrayDTypeMeta_associated_types,
                (PyObject *)&PyString_Type, (PyObject *)dtype_class);
        if (success < 0) {
            // TODO: Need to clean up in this unlikely event.
            return -1;
        }
    }
    else if (dtype_class->type_num == NPY_UNICODE) {
        success = PyDict_SetItem(
                PyArrayDTypeMeta_associated_types,
                (PyObject *)&PyUnicode_Type, (PyObject *)dtype_class);
        if (success < 0) {
            // TODO: Need to clean up in this unlikely event.
            return -1;
        }
    }

    dtype_class->dt_slots = dt_slots;

    dtype_class->dt_slots->default_descr = legacy_default_descr;
    dtype_class->dt_slots->discover_descr_from_pyobject =
                discover_descr_using_default;
    dtype_class->dt_slots->ensure_native = legacy_ensure_native;

    if (dtype_class->type_num == NPY_STRING ||
                    dtype_class->type_num == NPY_UNICODE) {
        dtype_class->dt_slots->discover_descr_from_pyobject =
                string_discover_descr_from_pyobject;
    }
    else if (dtype_class->type_num == NPY_DATETIME ||
                dtype_class->type_num == NPY_TIMEDELTA) {
        dtype_class->dt_slots->discover_descr_from_pyobject =
                discover_datetime_and_timedelta_from_pyobject;
    }
    else if (dtype_class->type_num == NPY_VOID) {
        dtype_class->dt_slots->discover_descr_from_pyobject =
                string_discover_descr_from_pyobject;
    }
    
    dtype_class->dt_slots->within_dtype_castingimpl = (
            (CastingImpl *)castingimpl_legacynew(dtype_class, dtype_class));
    if (dtype_class->dt_slots->within_dtype_castingimpl == NULL) {
        return -1;
    }

    dtype_class->dt_slots->can_cast_to_other = legacy_can_cast_to;
    dtype_class->dt_slots->can_cast_from_other = legacy_can_cast_from;

    if (dtype_class->type_num == NPY_OBJECT) {
        dtype_class->dt_slots->common_dtype = object_common_dtype;
    }
    else {
        dtype_class->dt_slots->common_dtype = legacy_common_dtype;
    }
    if (dtype_class->flexible) {
        /* Common instance is only necessary for flexible dtypes */
        if (dtype_class->type_num != NPY_VOID) {
            dtype_class->dt_slots->common_instance = legacy_common_instance;
        }
        else {
            /* VOID is not consistent with promotion vs. discovery */
            dtype_class->dt_slots->common_instance = VOID_common_instance;
        }
    }

    // This seems like it might make sense (but probably not here):
    Py_INCREF(dtype);
    //PyDict_SetItemString(dict, "sometype", dtype_classobj);

    return 0;
}


// TODO: Should really turn this around...
static CastingImpl *
can_cast_return_notimplemented(
            PyArray_DTypeMeta *NPY_UNUSED(cls),
            PyArray_DTypeMeta *NPY_UNUSED(other),
            NPY_CASTING NPY_UNUSED(casting)) {
    Py_INCREF(Py_NotImplemented);
    return (CastingImpl *)Py_NotImplemented;
}

static PyArray_DTypeMeta *
default_dtype_raise_no_default(PyArray_DTypeMeta *cls)
{
    PyErr_Format(PyExc_TypeError,
        "The abstract DType %s has no default concrete dtype.",
        (PyObject *)cls);
    return NULL;
}


/*NUMPY_API
 *
 * Initialize a DType subclasses DTypeMeta. Should be called after
 * PyType_Ready.
 */
int
PyArray_InitDTypeMetaFromSpec(
            PyArray_DTypeMeta *dtype_meta,
            PyArrayDTypeMeta_Spec *spec)
{
    PyType_Slot *slot;
    PyObject *associated_python_types = NULL;

    npy_bool is_abstract = spec->abstract;
    npy_bool is_flexible = spec->flexible;
    npy_intp itemsize = spec->itemsize;

    dtype_meta->typeobj = spec->typeobj;
    dtype_meta->abstract = is_abstract;
    dtype_meta->flexible = is_flexible;

    if (itemsize > NPY_MAX_INT) {
        PyErr_SetString(PyExc_RuntimeError, "itemsize must fit C-int.");
        return -1;
    }
    if (itemsize < 0) {
        assert(itemsize == -1);
    }
    if (is_abstract && itemsize != -1) {
        PyErr_SetString(PyExc_RuntimeError,
                "itemsize invalid for abstract DType (must be -1).");
        return -1;
    }

    if (Py_TYPE((PyObject *)dtype_meta) != &PyArrayDTypeMeta_Type &&
                Py_TYPE((PyObject *)dtype_meta) !=
                        &PyArrayAbstractObjDTypeMeta_Type) {
        /*
         * We can only check the first one, C Type definers have to allocate
         * enough space for the superclass.
         */
        PyErr_SetString(PyExc_RuntimeError,
            "A DType must be an instance of PyArrayDTypeMeta_Type and "
            "contain/be based on the PyArray_DTypeMeta struct.");
        return -1;
    }

    if (!is_abstract) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Currently only AbstractDTypes support!");
        return -1;
    }

    dtypemeta_slots *dt_slots = calloc(1, sizeof(dtypemeta_slots));
    dtype_meta->dt_slots = dt_slots;
    if (dtype_meta->dt_slots == NULL) {
        return -1;
    }

    for (slot = spec->slots; slot->slot; slot++) {
        if (slot->pfunc == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                "A given PyArray_InitDTypeMetaFromSpec slot must not be NULL.");
            goto fail;
        }
        // TODO: An offset array may be easier/nicer (less code).
        switch (slot->slot) {

        case NPY_dt_can_cast_from_other:
            dt_slots->can_cast_from_other = (can_cast_function *)slot->pfunc;
            continue;
        case NPY_dt_can_cast_to_other:
            dt_slots->can_cast_to_other = (can_cast_function *)slot->pfunc;
            continue;
        case NPY_dt_common_dtype:
            dt_slots->common_dtype = (common_dtype_function *)slot->pfunc;
            continue;
        case NPY_dt_common_instance:
            dt_slots->common_instance = (common_instance_function *)slot->pfunc;
            continue;
        case NPY_dt_default_dtype:
            dt_slots->default_dtype = (dtype_from_dtype_function *)slot->pfunc;
            continue;
        case NPY_dt_minimal_dtype:
            dt_slots->minimal_dtype = (dtype_from_dtype_function *)slot->pfunc;
            continue;
        // case NPY_dt_within_dtype_castingimpl:
        //     continue;
        case NPY_dt_legacy_arrfuncs:
            dtype_meta->f = slot->pfunc;
            continue;
        case NPY_dt_associated_python_types:
            associated_python_types = (PyObject *)slot->pfunc;
            Py_INCREF(associated_python_types);
            assert(Py_TYPE(associated_python_types) == &PyTuple_Type);
            continue;
        case NPY_dt_discover_dtype_from_pytype:
            dt_slots->discover_dtype_from_pytype =
                    (dtype_from_discovery_function *)slot->pfunc;
            continue;
            case NPY_discover_descr_from_pyobject:
            dt_slots->discover_descr_from_pyobject =
                    (descr_from_discovery_function *)slot->pfunc;
            continue;
        }
        PyErr_SetString(PyExc_RuntimeError, "invalid slot offset (or not yet implemented)");
        goto fail;
    }

    if (dt_slots->can_cast_from_other == NULL) {
        dt_slots->can_cast_from_other = &can_cast_return_notimplemented;
    }

    if (dt_slots->discover_descr_from_pyobject == NULL) {
        if (is_flexible) {
            PyErr_SetString(PyExc_RuntimeError,
                    "a flexible dtype must implement instance from pytype discovery.");
            goto fail;
        }
        dt_slots->discover_descr_from_pyobject = discover_descr_using_default;
    }

    if (spec->typeobj) {
        Py_INCREF(spec->typeobj);
        dtype_meta->typeobj = spec->typeobj;
    }
    else{
        Py_INCREF(Py_None);
        // TODO: Probably better store NULL...
        dtype_meta->typeobj = (PyTypeObject *)Py_None;
    }

    if (associated_python_types == NULL) {
        if ((PyObject *)dtype_meta->typeobj != Py_None) {
            associated_python_types = PyTuple_Pack(1, dtype_meta->typeobj);
        }
        else {
            associated_python_types = PyTuple_New(0);
        }
        if (associated_python_types == NULL) {
            goto fail;
        }
    }

    for (Py_ssize_t i = 0; i < PyTuple_Size(associated_python_types); i++) {
        // TODO: Not currently stored on the DType itself, maybe OK?
        int success;
        PyObject *typeobj = PyTuple_GetItem(associated_python_types, i);

        success = PyDict_Contains(PyArrayDTypeMeta_associated_types, typeobj);
        if (success < 0) {
            goto fail;
        }
        if (success) {
            // TODO: Fix handling here;
            printf("WARNING: DType to python type association already exists, ignoring!\n");
            continue;
        }
        success = PyDict_SetItem(
            PyArrayDTypeMeta_associated_types, typeobj, (PyObject *)dtype_meta);
        if (success < 0) {
            // TODO: Need to clean up in this unlikley event.
            goto fail;
        }
    }
    Py_DECREF(associated_python_types);
    associated_python_types = NULL;

    if (is_abstract) {
        /* Many slots must not be defined (or should never be used) */
        // TODO: Should ensure here, that `__new__` must fail/not be overwritten
        // TODO: Revise, add and probably make a macro...
        if (dt_slots->can_cast_to_other != NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                "AbstractDTypes can never be cast to another dtype.");
                goto fail;
        }
        if (dt_slots->common_instance != NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                "AbstractDTypes cannot define a common instance.");
                goto fail;
        }
        if (dt_slots->within_dtype_castingimpl != NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                "AbstractDTypes can never cast.");
                goto fail;
        }
        if (dt_slots->can_cast_to_other != NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                "AbstractDTypes can never be cast to another dtype.");
                goto fail;
        }

        if (dt_slots->default_dtype == NULL) {
            dt_slots->default_dtype = &default_dtype_raise_no_default;
        }
        if (dt_slots->minimal_dtype == NULL) {
            dt_slots->minimal_dtype = dt_slots->default_dtype;
        }

        return 0;
    }
    else {
        // Some things should not be implemented, such as default_dtype and
        // minimal_dtype.
        if (dt_slots->requires_pyobject_for_discovery ||
                    dt_slots->discover_dtype_from_pytype) {
            PyErr_SetString(PyExc_RuntimeError,
                    "Only AbstractDTypes may use dtype discovery functions.");
            goto fail;
        }
    }
    goto fail;

fail:
    Py_XDECREF(associated_python_types);

    Py_XDECREF(dtype_meta->typeobj);
    dtype_meta->typeobj = NULL;
    free(dt_slots);
    return -1;
}


NPY_NO_EXPORT PyTypeObject PyArrayDTypeMeta_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.DTypeMeta",
    .tp_basicsize = sizeof(PyArray_DTypeMeta),
    /* methods */
    .tp_dealloc = (destructor)dtypemeta_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "nonsense docs",
    // .tp_traverse = dtypemeta_traverse,
    // .tp_clear = dtypemeta_clear,
    .tp_members = dtypemeta_members,
    .tp_base = &PyType_Type,
    .tp_init = (initproc)dtypemeta_init,
    .tp_new = dtypemeta_new,
};


NPY_NO_EXPORT PyObject *PyArrayDTypeMeta_associated_types = NULL;