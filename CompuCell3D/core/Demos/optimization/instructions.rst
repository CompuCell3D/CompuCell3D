to run this demo you must use (dependeing on your architecture)
``optimization.bat``, ``optimization.sh`` or ``optimization.command`` run script


For example if you are on windows and have ``CompuCell3D installed`` in ``c:\CompuCell3D-64bit`` then
you would run the following command:

.. code-block:: console

    c:\CompuCell3D-64bit\optimization.bat --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d \
    --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json --cc3d-run-script=c:\CompuCell3D-64bit\runScript.bat \
    --clean-workdirs --num-workers=1 --population-size=6

or one-liner version of the above command:
.. code-block:: console

    c:\CompuCell3D-64bit\optimization.bat --input=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\optimization_demo.cc3d --params-file=c:\CompuCell3D-64bit\Demos\optimization\optimization_demo\Simulation\params.json --cc3d-run-script=c:\CompuCell3D-64bit\runScript.bat --clean-workdirs --num-workers=1 --population-size=6

**Notice** running of the optimization simulation from the player directly is
NOT supported and will result in an error.
