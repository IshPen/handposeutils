handposeutils.calculations package
==================================

Submodules
----------

handposeutils.calculations.geometry module
------------------------------------------

.. automodule:: handposeutils.calculations.geometry
   :members:
   :show-inheritance:
   :undoc-members:

vector_between(c1, c2)
~~~~~~~~~~~~~~~~~~~~~~

Returns a NumPy vector from Coordinate ``c1`` to ``c2``.

**Parameters:**

- ``c1`` (Coordinate): First point (x, y, z)
- ``c2`` (Coordinate): Second point (x, y, z)

**Returns:**

- ``np.array``: A NumPy array representing the vector from ``c1`` to ``c2``.

Example:

.. code-block:: python

    from handposeutils.calculations.geometry import vector_between
    v = vector_between(c1, c2)
    print(v)


handposeutils.calculations.similarity module
--------------------------------------------

.. automodule:: handposeutils.calculations.similarity
   :members:
   :show-inheritance:
   :undoc-members:

handposeutils.calculations.transforms module
--------------------------------------------

.. automodule:: handposeutils.calculations.transforms
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: handposeutils.calculations
   :members:
   :show-inheritance:
   :undoc-members:
