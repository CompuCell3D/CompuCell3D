# CompuCell3D Headless Docker Deployment

Created by Marcela Tarazona (marvita2023) and Michael Kowalski (MiKowalsk)

This program provides a headless deployment of CompuCell3D simulations using Docker, allowing execution of simulations with no GUI.

## Supported Simulation Types

We support two types of CompuCell3D simulations:

1. **Traditional CompuCell3D Simulations**: Consist of `.cc3d`, `.xml`, and `.py` files, with simulation fields available at/
   https://github.com/CompuCell3D/CompuCell3D/tree/master/CompuCell3D/core/Demos/CC3DPy/scripts
2. **Pure Python Simulations**: Simulation files can be found at/
   https://github.com/CompuCell3D/CompuCell3D/tree/master/CompuCell3D/DeveloperZone/Demos

The program includes 2 preloaded demo files for each type, totaling 4 demo files.

## Folder Structure

- `Simulation` folder: Contains simulation files.
- `Dockerfile`: Defines the Docker image setup.
- `ReadMe.markdown`: This file with instructions.
- Preloaded simulation files: `AdhesionDemo.py`, `ContactInternalDemo.py`, `CustomCellAttributesCpp.cc3d`, and `GrowthSteppable.cc3d`.

## Running Traditional CompuCell3D Simulations

Traditional simulations require `.cc3d`, `.xml`, and `.py` files. Follow these steps:

1. **File Placement**:

Clone CC3D repository:

```commandline
mkdir -p ~/src
cd ~/src
git clone git@github.com:CompuCell3D/CompuCell3D.git
```
The CC3D repository will be placed in ``~/src/CompuCell3D`` 

Next, copy folder with simulations to the ``~/src/CompuCell3D/docker``. For example, you may copy CC3D ``Demos``:

``
cp -r ~/src/CompuCell3D/CompuCell3D/core/Demos ~/src/CompuCell3D/docker
``

If your simulation is in a different folder you may need to adjust this path ``~/src/CompuCell3D/CompuCell3D/core/Demos``

2. **Build the Docker Image**:

 - Docker runs on all platforms but ideally you would run Docker on Linux because Docker is Linux-native app.
If you are on Windows you can always install linux under WSL2. It is probably better solution that running Docker 
purely under Windows

- Ensure Docker Desktop is running and installed.
- Run the following command in the terminal from the directory containing the `Dockerfile`:

  ```
  cd ~/src/CompuCell3D/docker
  docker build -t cc3d_cpu .
  ```
At this point the docker image with CC3D is ready for use

3. **Run the Simulation**:
The folder containing Docker file ``~/src/CompuCell3D/docker`` will be mounted inside Docker image as
``/sim``.  Therefore, if you want to run Simulation that is accessible on your computer as 
`` ~/src/CompuCell3D/docker/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d`` - remember that we copied the ``Demos``
folder to ``~/src/CompuCell3D/docker`` -  you would refer to this simulation file as ``/sim/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d``
Why? Because  ``~/src/CompuCell3D/docker`` on your computer becomes ``/sim`` inside docker image.

Therefore, to run a simulation you run the following:

```commandline
cd ~/src/CompuCell3D/docker
docker run -it --rm -v "${PWD}:/sim" cc3d_cpu python -m cc3d.run_script -i /sim/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d -o /sim/cellsort_results -f 1000
```

Notice , we specify output folder to be ``/sim/cellsort_results``. This means that inside the container the simulation output results will be written to
``/sim/cellsort_results`` but because ``/sim`` is accessible on your host machine as ``~/src/CompuCell3D/docker`` therefore on your local machine you may inspect ``~/src/CompuCell3D/docker/cellsort_results`` and you will find thet indeed all simulations' results are in your local folder - ``~/src/CompuCell3D/docker/cellsort_results``. 


## Running Pure Python CompuCell3D Simulations

Pure Python simulations require the necessary `.py` files. and you follow the steps outlined above with some key exceptions:

1. **File Preparation**:

   - Place the `.py` file outside the `Simulation` folder (e.g., at the root level).
   - Comment out visualization-related code (e.g., `cc3d_sim.visualize()` or input prompts like `input()`).
   - Keep only the following execution loop:

     ```python
     while cc3d_sim.current_step < 10000:
     
         cc3d_sim.step()
     print(cc3d_sim.profiler_report)
     ```

2. **Build the Docker Image**:

   - Ensure Docker Desktop is running and installed.
   - Run the following command in the terminal from the directory containing the `Dockerfile`:

     ```
     docker build -t cc3d_cpu .
     ```

3. **Run the Simulation**:

    Assuming that your Python only simulations are places inside ``~/src/CompuCell3D/docker/python_only`` folder , for example,   ``~/src/CompuCell3D/docker/python_only/AdhesionDemo.py`` or 
``~/src/CompuCell3D/docker/python_only/ContactInternalDemo.py`` you can use  the appropriate command based on the simulation file:

     ```
     docker run --rm -v "${PWD}:/sim" cc3d_cpu python /sim/python_only/AdhesionDemo.py --output-dir=/sim/adhesion_demo_results
     ```

     or

     ```
     docker run --rm -v "${PWD}:/sim" cc3d_cpu python /sim/python_only/ContactInternalDemo.py --output-dir=/sim/contact_internal_results
     ```
All results will be visible on your host computer inside ``~/src/CompuCell3D/docker/`` folder

## Notes

- The Docker image is tagged as `cc3d_cpu` for consistency.
- Ensure all necessary simulation files are correctly placed before building and running the image.