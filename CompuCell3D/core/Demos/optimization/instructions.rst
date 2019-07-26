This ia simple demo that illustrates how to implement parameter optimization in CC3D

.. note::

    The objective function we minimize is a simple quadratic form involving two parameters
    It doe not depend on the simulation output hence in fact we run simple quadratic
    function optimization where each optimization step requires run of CC3D simulation with
    those new parameters. You can extend this demo and replace our simple objective function
    with a function that depends on the actual simulation output for example the length of
    heterotypic boundary between two cell types.

To run this demo you must use (depending on your architecture)
``optimization.bat``, ``optimization.sh`` or ``optimization.command`` run script


For example if you are on windows and have ``CompuCell3D installed`` in ``c:\CompuCell3D-64bit`` then
you would run the following command:

.. code-block:: console

    c:\CompuCell3D-64bit\optimization.bat --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d \
    --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json \
    --cc3d-run-script=c:\CompuCell3D-64bit\runScript.bat \
    --clean-workdirs --num-workers=1 --population-size=6

or one-liner version of the above command:
.. code-block:: console

    c:\CompuCell3D-64bit\optimization.bat --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json --cc3d-run-script=c:\CompuCell3D-64bit\runScript.bat --clean-workdirs --num-workers=1 --population-size=6

**Notice** running of the optimization simulation from the player directly is
NOT supported and will result in an error.

Optimization with GUI output
============================

If you want GUI output while running simulation, all you need to do is to to change path top the run script to point
to the ``Player`` run script (in our case we would use ``--cc3d-run-script=c:\CompuCell3D-64bit\compucell3d.bat``)

.. note::

    Before running optimization with GUI you must check ``Close Player after simulation ends``
    in the Player configuration  dialog. Without this step the optimization run
    in the GUI mode will not work properly

.. code-block:: console

    c:\CompuCell3D-64bit\optimization.bat --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json --cc3d-run-script=c:\CompuCell3D-64bit\compucell3d.bat --clean-workdirs --num-workers=1 --population-size=6

.. warning::

    Even though GUI runs are possible if you use more than one worker you will end up with
    multiple GUIs being open at the same time. also the memory usage wil spike so this is
    something you should be aware of when using multiple workers with a GUI-base
    optimization runs


"d:\Program Files\3710\optimization.bat" --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json --cc3d-run-script="d:\Program Files\3710\runScript.bat" --clean-workdirs --num-workers=1 --population-size=6