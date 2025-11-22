CompuCell3D
===========

CompuCell3D is a multiscale multicellular virtual tissue modeling and simulation environment.
CompuCell3D is written in C++ and provides Python bindings for model and simulation development
in Python. CompuCell3D is supported on Windows, Mac and Linux.

Website
-------
`CompuCell3D project page <https://compucell3d.org/>`_

CompuCell3D Community
----------------------------

For bug reports, feature, requests and COmpuCel3D discussions we encourage you to visit `CompuCell3D Community Guide <https://github.com/CompuCell3D/CompuCell3D/blob/master/README_CompuCell3D_Community.md>`_ 

Installation
------------

Binaries
********

Binary distributions of CompuCell3D are available for download at the
`CompuCell3D project page <https://compucell3d.org/SrcBin>`_. Binaries of CompuCell3D are
also available via conda,

.. code-block:: console

    conda install -c compucell3d -c conda-forge cc3d

Once installation completes, you may run Player (CompuCell3D GUI) as follows:

.. code-block:: console

    python -m cc3d.player5

To run Model editor (Twedit++)  you would type:

.. code-block:: console

    python -m cc3d.twedit5


Finalyy, if you want to run simulation in a GUI-less mode you run:

.. code-block:: console

    python -m cc3d.run_script -i <full path to .cc3d file>




Source
******

Instructions for building CompuCell3D from source are available at the
`CompuCell3D project page <https://compucell3d.org/>`_.

Getting Started
---------------

- Model development for CompuCell3D is best performed using the
  `Twedit++ IDE <https://github.com/CompuCell3D/cc3d-twedit5/tree/master>`_.

- CompuCell3D simulations can be interactively executed with real-time rendering using
  `CompuCell3D Player <https://github.com/CompuCell3D/cc3d-player5/tree/master>`_.

- Core CompuCell3D modeling and simulation features are documented in the
  `CompuCell3D Reference Manual <https://compucell3dreferencemanual.readthedocs.io/en/latest/index.html>`_.

- CompuCell3D Python model and simulation development and walkthroughs are documented in the
  `Python Scripting Manual for CompuCell3D <https://pythonscriptingmanual.readthedocs.io/en/latest/index.html>`_.

- CompuCell3D is distributed with an extensive
  `repository of demos <https://github.com/CompuCell3D/CompuCell3D/tree/master/CompuCell3D/core/Demos>`_.

- `CompuCell3D Extensions <https://github.com/CompuCell3D/CompuCell3DExtensions/tree/main>`_
  offer integrated third-party modeling and simulation capabilities.
