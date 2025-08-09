# CompuCell3D Headless Docker Deployment

Created by Marcela Tarazona (marvita2023) and Michael Kowalski (MiKowalsk)

This program provides a headless deployment of CompuCell3D simulations using Docker, allowing execution of simulations with no GUI.

## Supported Simulation Types

We support two types of CompuCell3D simulations:

1. **Traditional CompuCell3D Simulations**: Consist of `.cc3d`, `.xml`, and `.py` files, with simulation fields available at\
   https://github.com/CompuCell3D/CompuCell3D/tree/master/CompuCell3D/core/Demos/CC3DPy/scripts
2. **Pure Python Simulations**: Simulation files can be found at\
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

   - Place the `.py` and `.xml` files in the `Simulation` folder.
   - Place the corresponding `.cc3d` file outside the `Simulation` folder (e.g., at the root level).
   - **Note**: Since this is a headless deployment, comment out any `<Steppable Type="###">` lines (where `###` is the simulation name) in the `.xml` files to ensure compatibility.

2. **Build the Docker Image**:

   - Ensure Docker Desktop is running and installed.
   - Run the following command in the terminal from the directory containing the `Dockerfile`:

     ```
     docker build -t cc3d_cpu .
     ```

3. **Run the Simulation**:

   - Use the appropriate command based on the simulation file:

     ```
     docker run -it --rm -v "${PWD}:/sim" cc3d_cpu python -m cc3d.run_script -i /sim/CustomCellAttributesCpp.cc3d
     ```

     or

     ```
     docker run -it --rm -v "${PWD}:/sim" cc3d_cpu python -m cc3d.run_script -i /sim/GrowthSteppable.cc3d
     ```

## Running Pure Python CompuCell3D Simulations

Pure Python simulations require the necessary `.py` files. Follow these steps:

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

   - Use the appropriate command based on the simulation file:

     ```
     docker run --rm -v "${PWD}:/sim" cc3d_cpu conda run -n cc3d_env python /sim/AdhesionDemo.py
     ```

     or

     ```
     docker run --rm -v "${PWD}:/sim" cc3d_cpu conda run -n cc3d_env python /sim/ContactInternalDemo.py
     ```

## Notes

- The Docker image is tagged as `cc3d_cpu` for consistency.
- Ensure all necessary simulation files are correctly placed before building and running the image.