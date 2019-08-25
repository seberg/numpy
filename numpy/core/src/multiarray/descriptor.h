#ifndef _NPY_ARRAYDESCR_H_
#define _NPY_ARRAYDESCR_H_

NPY_NO_EXPORT PyObject *arraydescr_protocol_typestr_get(PyArray_Descr *);
NPY_NO_EXPORT PyObject *arraydescr_protocol_descr_get(PyArray_Descr *self);

NPY_NO_EXPORT PyObject *
array_set_typeDict(PyObject *NPY_UNUSED(ignored), PyObject *args);

int
_arraydescr_from_dtype_attr(PyObject *obj, PyArray_Descr **newdescr);


NPY_NO_EXPORT int
is_dtype_struct_simple_unaligned_layout(PyArray_Descr *dtype);

/*
 * Filter the fields of a dtype to only those in the list of strings, ind.
 *
 * No type checking is performed on the input.
 *
 * Raises:
 *   ValueError - if a field is repeated
 *   KeyError - if an invalid field name (or any field title) is used
 */
NPY_NO_EXPORT PyArray_Descr *
arraydescr_field_subset_view(PyArray_Descr *self, PyObject *ind);

extern NPY_NO_EXPORT char *_datetime_strings[];

/*
 * Slots of DTypeMeta, Probably can use the same structure for AbstractDTypeMeta.
 * This must remain be fully opaque!
 */
typedef struct {
        PyObject_HEAD
        /*
         * the type object representing an
         * instance of this type -- should not
         * be two type_numbers with the same type
         * object.
         */
        PyTypeObject *typeobj;
        /* kind for this type */
        char kind;
        /* unique-character representing this type */
        char type;
        /*
         * '>' (big), '<' (little), '|'
         * (not-applicable), or '=' (native).
         */
        /* flags describing data type */
        char flags;
        npy_bool flexible;
        npy_bool abstract;
        /* number representing this type */
        int type_num;
        /* element size (itemsize) for this type, can be -1 if flexible. */
        int elsize;  // TODO: Think about making it intp? How much API would actually be broken by this?
        /* alignment needed for this type */
        // int alignment;   Maybe add again?
        /*
         * Link to the original f, should be removed at some point probably.
         * Maybe this could become a copy, just to know if something happened
         * in the meantime.
         */
        PyArray_ArrFuncs *f;
} PyArray_DTypeMeta;


#endif
