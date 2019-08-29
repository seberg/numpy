#ifndef _NPY_DTYPEMETA_H_
#define _NPY_DTYPEMETA_H_

NPY_NO_EXPORT int descr_dtypesubclass_init(PyArray_Descr *dtype);


struct _CastingImpl;
struct _PyArray_DTypeMeta;

typedef PyArray_Descr *(default_descr_func)(struct _PyArray_DTypeMeta *cls);

typedef struct _PyArray_DTypeMeta *(common_dtype_function)(
            struct _PyArray_DTypeMeta *cls,
            struct _PyArray_DTypeMeta *other);

typedef PyArray_Descr *(common_instance_function)(
            struct _PyArray_DTypeMeta *cls,
            PyArray_Descr *descr1, PyArray_Descr *descr2);

typedef struct _CastingImpl *(can_cast_function)(
            struct _PyArray_DTypeMeta *cls,
            struct _PyArray_DTypeMeta *other,
            NPY_CASTING casting);

/*
 * This struct must remain fully opaque to the user, direct access is
 * solely allowed from within NumPy!
 * (Others have to use a PyArrayDType_GetSlot(), which may return an error
 * or a similar name).
 * Not all slots will be exposed.
 */
typedef struct {
    /* PyObject handling: */
    void *getitem;
    void *setitem;
    void *discover_dtype_from_pytype;
    /* Casting: */
    can_cast_function *can_cast_from_other;
    can_cast_function *can_cast_to_other;
    common_dtype_function *common_dtype;
    common_instance_function *common_instance;
    default_descr_func *default_descr;
    /* Slots for legacy wrapper */
    //PyObject *legacy_castingimpls_from;
    //PyObject *legacy_castingimpls_to;
    //int *legacy_casting_from_info;
    //int *legacy_casting_to_info;
    /* Special slots */
    struct _CastingImpl *within_dtype_castingimpl;
} dtypemeta_slots;


/*
 * Slots of DTypeMeta, Probably can use the same structure for AbstractDTypeMeta.
 * This must remain be fully opaque!
 */
typedef struct _PyArray_DTypeMeta {
        // NOTE: This is allocated as PyHeapTypeObject, but most dtypes do not
        //       actually require that. Value based casting should though, and
        //       downstream should have the ability. (I hope this does not get difficult :/)
        PyHeapTypeObject super;
        // TODO: I want these slots to be just 2-3 pointers, i.e.
        // int abstract
        // dtypemeta_slots *dt_slots  /* Private growable struct */
        // This means that downstream can rely on the size of the supertype
        // in an ABI compatible manner while we can extend our API freely.
        
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
        npy_bool is_legacy_wrapper;
        // Most things should go into this single pointer, so that things
        // are nice and clean and hidden away:
        dtypemeta_slots *dt_slots;
} PyArray_DTypeMeta;

#endif  /* _NPY_DTYPEMETA_H_ */
