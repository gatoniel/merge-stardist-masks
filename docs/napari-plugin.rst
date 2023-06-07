napari plugin
=============


Installation
------------

Make sure to have a running `napari installation`_.

Via pip
^^^^^^^

In the environment with your napari installation run:

.. code:: console

   $ pip install merge-stardist-masks


Within napari
^^^^^^^^^^^^^

Within napari go to ``Plugins -> Install/Uninstall Plugins...`` and search for
*napari-merge-stardist-masks* in the lower list. Then click on the blue *install*
button.

After installation
^^^^^^^^^^^^^^^^^^

Make sure to restart napari after the installation. If you do not find the plugins, go
to ``Plugins -> Install/Uninstall Plugins...`` and toggle the checkboxes in the upper list for
*stardist-napari* and *napari-merge-stardist-masks*.


Usage
-----

Preparations
^^^^^^^^^^^^

Download one of the pre-trained StarDist models from `here`_ and unzip the
file.

Run a segmentation
^^^^^^^^^^^^^^^^^^

1. Load sample data with ``File -> Open sample -> StarDist OPP sample data``
2. Click ``Plugins -> StarDist OPP``. Two widgets will open, the `StarDist plugin`_ and this plugin.
3. All the parameters in the *StarDist plugin* should be correctly set already. Make sure that the axes in the field ``Image Axes`` are correct, for a 3D image it should be ``ZYX``.
4. Select ``Custom 2D/3D`` in the field ``Model Type`` and choose the directory where you unzipped the pre-trained model in the ``Custom Model`` field. See the image below for the correct settings.

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/paper/images/stardist-widget.png

5. Hit ``Run``. And wait until the CNN calculates the outputs. The outputs of the CNN are displayed once they are calculated.
6. In the *StarDist OPP* widget, select again the path to the unzipped pre-trained model in the field ``model path``. Then select ``StarDist distances (data)`` and ``StarDist probability (data)`` for the ``dists`` and ``probs`` fields, respectively.
7. You can play around with the other fields. However, this might lead to errors. For 3D images, you should set ``subtract dist`` to ``1.00`` the other settings are already fine. See the following image for proper settings.

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/paper/images/stardist-opp-widget.png

8. Hit ``Run`` in the *StarDist OPP* widget. The post-processing starts and might take some time (on our machine it takes ~10 minutes). Once the post-processing is done, the label image will be shown in the viewer.

.. image:: https://github.com/gatoniel/merge-stardist-masks/raw/paper/images/final.png


.. _napari installation: https://napari.org/stable/tutorials/fundamentals/installation.html
.. _here: https://github.com/gatoniel/napari-merge-stardist-masks/tree/main/models
.. _StarDist plugin: https://github.com/stardist/stardist-napari
