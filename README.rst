StarDist OPP
============

|PyPI| |Zenodo| |Status| |Python Version| |License|

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
.. |Zenodo| image:: https://zenodo.org/badge/454865222.svg
   :target: https://zenodo.org/badge/latestdoi/454865222
   :alt: Zenodo


This repository contains the python package for the new StarDist post-processing step *StarDist OPP*. *StarDist OPP* allows to use `StarDist`_ segmentation on non-star-convex objects. In our `paper`_, we show that *StarDist OPP* outperforms other methods in instance segmentation tasks for three-dimensional microbial biofilms. Check it out for more information.

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/main/images/graphical-overview.png


Features
--------

* *StarDist OPP* merges masks together - hence the repository name

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/main/images/graphical-algorithm-explanation.png

* *StarDist OPP* works in 2D and 3D

* In 2D, *StarDist OPP* works also on big and winding objects


Requirements
------------

* A `StarDist`_ installation.


Usage
-----

Please see the EXAMPLE in `Usage <Usage_>`_ for details or check out the `tutorial`_ of our `napari plugin`_ to directly use *StarDist OPP* on your data.


Installation
------------

You can install *StarDist OPP* via pip_ from PyPI_:

.. code:: console

   $ pip install merge-stardist-masks


Contributing
------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `MIT license`_,
*StarDist OPP* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


How to cite
-----------

.. code:: bibtex

   @article{https://doi.org/10.1111/mmi.15064,
   author = {Jelli, Eric and Ohmura, Takuya and Netter, Niklas and Abt, Martin and Jiménez-Siebert, Eva and Neuhaus, Konstantin and Rode, Daniel K. H. and Nadell, Carey D. and Drescher, Knut},
   title = {Single-cell segmentation in bacterial biofilms with an optimized deep learning method enables tracking of cell lineages and measurements of growth rates},
   journal = {Molecular Microbiology},
   volume = {119},
   number = {6},
   pages = {659-676},
   keywords = {3D segmentation, biofilm, deep learning, image analysis, image cytometry, Vibrio cholerae},
   doi = {https://doi.org/10.1111/mmi.15064},
   url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/mmi.15064},
   eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/mmi.15064},
   abstract = {Abstract Bacteria often grow into matrix-encased three-dimensional (3D) biofilm communities, which can be imaged at cellular resolution using confocal microscopy. From these 3D images, measurements of single-cell properties with high spatiotemporal resolution are required to investigate cellular heterogeneity and dynamical processes inside biofilms. However, the required measurements rely on the automated segmentation of bacterial cells in 3D images, which is a technical challenge. To improve the accuracy of single-cell segmentation in 3D biofilms, we first evaluated recent classical and deep learning segmentation algorithms. We then extended StarDist, a state-of-the-art deep learning algorithm, by optimizing the post-processing for bacteria, which resulted in the most accurate segmentation results for biofilms among all investigated algorithms. To generate the large 3D training dataset required for deep learning, we developed an iterative process of automated segmentation followed by semi-manual correction, resulting in >18,000 annotated Vibrio cholerae cells in 3D images. We demonstrate that this large training dataset and the neural network with optimized post-processing yield accurate segmentation results for biofilms of different species and on biofilm images from different microscopes. Finally, we used the accurate single-cell segmentation results to track cell lineages in biofilms and to perform spatiotemporal measurements of single-cell growth rates during biofilm development.},
   year = {2023}
   }

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/main/images/stardist-opp-cover-image.png


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
.. _paper: https://doi.org/10.1111/mmi.15064
.. _napari plugin: https://github.com/gatoniel/napari-merge-stardist-masks
.. _tutorial: https://merge-stardist-masks.readthedocs.io/en/latest/napari-plugin.html
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://merge-stardist-masks.readthedocs.io/en/latest/usage.html
