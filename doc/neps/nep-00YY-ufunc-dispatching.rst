===================================
NEP XX — Improved UFunc Dispatching
===================================

:title: Extensible Datatypes for NumPy
:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2020-01-17


Abstract
--------

In NEP 34 the need for new user defined datatypes was motivated.
Allowing users to create more powerful datatypes is however only one
side of the coin.
Any dtype can only be useful if it can most functions can be executed on it.
Within NumPy, this means that universal functions (*UFuncs*), which implement
almost all elementwise and many non-elementwise functions, must be able
to operate on all dtypes.
As of now, this is however limited in several ways and, for datatypes such
as strings or datetimes is only (reasonably) possible from within NumPy.
Even for NumPy itself, however, the solution is clumsy and obscure enough
that currently no UFunc exists which can handle the string dtype.
Universal functions work by having multiple possible *loops* or *implementations*
for various dtypes definied.
But, often the user input will match none of the loops exactly leading to the
need of *promotion* before the actual loop is chosen (*dispatching*).
The way promotion is handled in NumPy is a further shortcoming that should
be alleviated.


Detailed Description
--------------------

UFuncs are implemented as a main object handling most of the heavy lifting,
which calls then into *implementations* which are specific to an exact DType
class type signature.
Currently, the implementation consists mainly of a single C-function
implementing a simple loop and the actual necessary operation.
Each implementation is thus specific to specific, exact input and output
dtypes.
For NumPy dtypes, the available implementations/loops can be listed 
using ``ufunc.types`` (e.g. ``np.add.types``).

After initial steps, such as checking whether the ``__array_ufunc__`` protocol
should be used, UFunc execution has to perform multiple steps:

1. Find the correct *implementation*, which means finding the DTypes
   used during the operation. (Note that these are the DType classes, which
   are currently the type numbers)
2. Find the correct dtype instances to be used. This is necessary for setting
   up the iteration (including buffers for possible casting),
   as well as allocating the result array.
3. Prepare any additional data which is to be passed into the implementation
   specific code. (Currently, this step is very limited)
4. Perform the loop and call the implementations inner-loop function as often
   as necessary.
5. Check whether an error occured. (Currently, generic code for all
   implementations)

Note that currently the *implementation* mainly consists of a single C function
pointer.

The main difficulty addressed in this NEP is the first, *dispatching* step,
which may need to involve some type of *promotion*.


1 – Dispatching and Promotion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a user for example adds a floating point number and an integer, numpy
will automatically decide that the floating point number will be used.
This promotion is important for many use cases.
The way this is currently done is that 

