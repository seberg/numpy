Compatibility breaking changes in UFuncs:

  * The out argument now insures a high precision loop
    will be picked. See also: https://mail.python.org/pipermail/numpy-discussion/2019-September/080106.html
  * New style registered loops will be preferred.
    This could cause conflicts if multiple downstream libraries
    register loops for void types (new style). This should be
    discouraged!
    Void type loops registered within numpy, should be generic
    enough to never clash (the main/only candidate being equal
    which is well defined, assuming metadata is not used).
  * Customizing ``masked_inner_loop_selector`` will not be supported anymore!
    (Optimized masked loops may follow later.)
