============================================================
NEP 51 — Python integer conversion and default integer dtype
============================================================
:Author: Sebastian Berg
:Status: Draft
:Type: Standards Track
:Created: 2021-03-24


Abstract
========

When converting a Python integer to a NumPy array or scalar using for example::

    np.array([1000])

NumPy has to decide which ``dtype`` to use for the result.  Previously,
NumPy does this by trying the following dtypes and checking whether they
can hold the value::

    long -> int64 -> uint64 -> object

Where ``long`` may be ``int32`` or ``int64`` depending on the platform.
This leads two surprises.  First, the default integer depends on the platform
due to the use of C ``long``.  Second, the resulting ``dtype`` can be
``unsigned`` or ``object`` for large values.

To remove both surprises, we propose that NumPy always convert Python
integers to ``int32`` on 32-bit system and ``int64`` on 64-bit systems.
When the value is too large, an ``OverflowError`` will be raised.
This will change the default integer dtype which is also the minimum precision
returned by ``np.sum`` and ``np.prod``.

.. note::

    The proposed change here is distinct, but related to the proposal that
    ``np.array([1]) + 100`` should not inspect inspect the value of the ``100``
    to decide the resulting dtype.  This is proposal is discussed in NEP 50.


Motivation and Scope
====================

The current scheme of trying the the dtypes::

    long -> int64 -> uint64 -> object

when converting a Python integer e.g. using ``np.array(1000)``,
leads to two surprises that users and library authors can run into:

1. The use of ``long`` means that 64bit Windows work by default work often
   with ``int32``, while other operating systems will use ``in64``.
   This makes the default integer depend on the platform in a non-obvious way
   which may lead to incorrect results for software tested on linux,
   but running on Windows systems.
2. The fact that we try ``uint64`` and ``object`` means that the return dtype
   for many calls may be surprising for very large integers.

We can give consistent and predictable results by making ``np.intp`` the
"default integer" and never choosing a bigger integer.

The goal is thus to give consistent results: 32bit integers on 32bit platforms
and 64bit integers on 64bit platforms.

Previously, NumPy could have given ``uint64`` or ``object`` dtype which
behave differently from a typical integer.
By ensuring an ``OverflowError`` is raised in these cases, NumPy users who
deal with large integers will not be surprised by behaviour such as::

    >>> np.array([2**63])
    array([9223372036854775808], dtype=uint64)
    >>> np.array([1, 2**63])
    array([2.00000000e+00, 9.22337204e+18])

Where the first goes to ``uint64`` due to the size of the integer and the
second additionally confuses because ``int64`` and ``uint64`` promote to
``float64`` (which is partially a distinct problem).

Further, while the the ``object`` dtype can be convenient when dealing with
large integers, users who work with numbers around ``2**62`` may sometimes
get Python's arbitrary precision values and other times NumPy's limited
precision.
That behaviour seems more dangerous than the need of passing ``dtype=object``
explicitly.


Usage and Impact
================

In general, we believe that the better predictibility is a net gain for users.
The rule that the default integer is 32bit on 32bit systems and 64bit on 64bit
systems is much easier than the current one (which requires C knowledge to
explain).

When working with Windows 64bit, some libraries will have lingering bugs
for very large integers or arrays.
It is a fairly common mistake to see casts to ``dtype=int`` for arrays that
will later be used for indexing purposes.
However, on 64bit Windows this is *incorrect* and may fail *silently* when
working with very large arrays!

A cast to ``dtype=int`` (or later calculations) will wrap-around at
``2**31-1`` and an indexing operation may operate on the wrong elements.

In general, we feel the new proposed behaviour is both much safer and easier
to learn.
As an additional advantage, many NumPy indexing related operations are expected
to speed up on 64bit Windows.


Backward compatibility
======================

The main backwards compatibility concern is potential doubling of memory
use for some software targeted to Windows only.

For 64bit Windows users
-----------------------

Users of 64bit Windows specifically will now often work with 64bit integers
instead of 32bit integers.
In general we expect the increased precision to be better for the average user
since users who wish to trade precision for speed can use ``dtype=int32``
explicitly.

However, ignorig small reductions in performance, there are two failure paths
where users may have backwards compatibility concerns:
1. For memory intensive tasks this change may lead to doubling the memory use
   in some cases.  Potentially using too much memory or causing a large loss of
   performance.
2. In principle a user could rely on integer overflow behaviour but not
   pass ``dtype=int32`` explicitly.

We expect that the second point is so rare that if it exists the up-side of
fixing existing bugs for very large arrays outweighs the downside.

However, some authors of Windows specific software are expected to experience
memory bloat that will require update of their code.
These users will have to add ``dtype=np.int32`` explicitly in certain
operations.


For users of large integers
---------------------------

The proposed change will mean that some users will not automatically get
``uint64`` or ``object`` arrays when the previously did.
This means that in rare cases users may have to use logic such as::

    l = [1, 2**100]
    try:
        np.array(l)
    except OverflowError:
        np.array(l, dtype=object)

In theory, this may be inconvenient for rare use-cases where the
``OverflowError`` could have originated from elsewhere.
In practice, we currently assume that most users of very large integers will
appreciate being informed that ``dtype=object`` is used.


Related Work
============



Implementation
==============

The implementation consists of two steps:

1. The change of the default integer to ``np.intp`` in the C-level to ensure
   that ``np.sum`` and ``np.prod`` will use the higher precision.
2. The simplification of the logic in ``discover_descriptor_from_pyint`` to
   always choose an ``intp`` compatible result (or raise).
   (There may be a duplication of this logic elsewhere.)


Alternatives
============

Besides keeping the old behaviour the main alternative would be to always
use ``int64`` as the default integer dtype rather than the rule that 32bit
systems use 32bit integers and 64bit systems use 64bit integers.

The rule here was chosen for three reasons:
1. It aligns with existing for functions that return integers that are ready
   for indexing, such as `numpy.nonzero`.
2. It affects fewer users since it only applies to 64bit Windows in practice.

In practice, 32bit systems are becoming less typical and we expect it to be
much less surprising to see a difference in the default integer precision for
32bit systems.

In principle, the change here does not preclude a general change to 64bit.
Such a general change is possible, but it would seem necessary to also change
the precision of functions like `numpy.nonzero` to prefer 64bit integers
when 32bit integers are sufficient.



Discussion
==========



References and Footnotes
========================


Copyright
=========

This document has been placed in the public domain.
