The following incompatible changes occurred due to a new DType system

* The Macro PyArray_DescrCheck is invalid if compiled with an old version
  of NumPy. XXXX: An updated old version of NumPy can fix this.
* Identically due to the macro above ``type(dtype) is np.dtype`` checks will
  now fail, use ``isinstance(dtype, np.dtype)`` instead.
* Coercing an object with an invalid ``__array_interface__`` will now raise
  an error always, instead of creating an object array.
