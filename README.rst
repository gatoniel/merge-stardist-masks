Merge StarDist Masks
====================

|PyPI| |Status| |Python Version| |License|

|Read the Docs| |Tests| |Codecov|

|pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/merge-stardist-masks.svg
   :target: https://pypi.org/project/merge-stardist-masks/
   :alt: PyPI
.. |Status| image:: https://img.shields.io/pypi/status/merge-stardist-masks.svg
   :target: https://pypi.org/project/merge-stardist-masks/
   :alt: Status
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/merge-stardist-masks
   :target: https://pypi.org/project/merge-stardist-masks
   :alt: Python Version
.. |License| image:: https://img.shields.io/pypi/l/merge-stardist-masks
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/merge-stardist-masks/latest.svg?label=Read%20the%20Docs
   :target: https://merge-stardist-masks.readthedocs.io/
   :alt: Read the documentation at https://merge-stardist-masks.readthedocs.io/
.. |Tests| image:: https://github.com/gatoniel/merge-stardist-masks/workflows/Tests/badge.svg
   :target: https://github.com/gatoniel/merge-stardist-masks/actions?workflow=Tests
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/gatoniel/merge-stardist-masks/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/gatoniel/merge-stardist-masks
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

* This new post-processing step allows to use `StarDist`_ segmentation on
  non-star-convex objects.

  * Instead of NMS, this post-processing naively merges masks together

  * Masks whos center points lie within another mask are added to that mask

* Works in 2D and 3D

* In 2D, it works on big and winding objects


Requirements
------------

* A `StarDist`_ installation.


Installation
------------

You can install *Merge StarDist Masks* via pip_ from PyPI_:

.. code:: console

   $ pip install merge-stardist-masks


Usage
-----

Please see the EXAMPLE in `Usage <Usage_>`_ for details.


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `MIT license`_,
*Merge StarDist Masks* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


Credits
-------

This project was generated from `@cjolowicz`_'s `Hypermodern Python Cookiecutter`_ template.

.. _@cjolowicz: https://github.com/cjolowicz
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT license: https://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern Python Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _file an issue: https://github.com/gatoniel/merge-stardist-masks/issues
.. _pip: https://pip.pypa.io/
.. _StarDist: https://github.com/stardist/stardist
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://merge-stardist-masks.readthedocs.io/en/latest/usage.html
