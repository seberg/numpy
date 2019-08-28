#ifndef _NPY_CASTINGIMPL_H_
#define _NPY_CASTINGIMPL_H_

/*
 * This struct requires a rehaul, it must be compatible with the UFuncImpl
 * one. And should be extensible (and user subclassable) in the future.
 * This means that the actual public API should be short and possibly shared
 * with UFuncImpl.
 * Even if it is short enough to fully define it, we should add at least
 * one reserved slot which can point to a dynamically allocated (and opaque)
 * layer similar to the DTypeMeta field.
 */
typedef struct {
    void *adjust_descriptors;
    void *get_transferfunction;
} CastingImpl;


#endif  /* _NPY_CASTINGIMPL_H_ */
