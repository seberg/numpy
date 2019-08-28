#ifndef _NPY_DTYPEMETA_H_
#define _NPY_DTYPEMETA_H_

NPY_NO_EXPORT int descr_dtypesubclass_init(PyArray_Descr *dtype);


typedef struct {
    /* PyObject handling: */
    void *getitem;
    void *setitem;
    void *discover_dtype_from_pytype;
    /* Casting: */
    void *can_cast_from_other;
    void *can_cast_to_other;
    void *common_dtype;
    void *default_descr;
} dtypemeta_slots;


/*
 * Slots of DTypeMeta, Probably can use the same structure for AbstractDTypeMeta.
 * This must remain be fully opaque!
 */
typedef struct {
        PyTypeObject super;
        // TODO: I want these slots to be just 2-3 pointers, i.e.
        // int abstract
        // void *tp_descrslots  /* Private growable struct */
        // This means that downstream can rely on the size of the supertype
        // in an ABI compatible manner.
        // The struct would be heap allocated.
        
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
        /* flags describing data type */
        char flags;
        int flexible;
        int abstract;
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
        PyObject *name;
        // Most things should go into this single pointer, so that things
        // are nice and clean and hidden away:
        dtypemeta_slots *dt_slots;
} PyArray_DTypeMeta;

#endif  /* _NPY_DTYPEMETA_H_ */
