Changelog
=========

1.3.1 (2026-11-23)
------------------

Fixed
^^^^^
- Improve deep extraction for list-like columns containing scalar values.
- Handle Python and NumPy scalar types (`str`, `int`, `float`, `bool`,
  `np.integer`, `np.floating`, `np.bool_`) in `_col2type`.
- Harden `_get_unique_elements` to safely process unexpected non-list
  leftovers without failing extraction.

Tests
^^^^^
- Add regression tests for deep extraction with NumPy scalar values in
  list-typed columns.
- Add regression tests for deep extraction with boolean scalar values in
  list-typed columns.
