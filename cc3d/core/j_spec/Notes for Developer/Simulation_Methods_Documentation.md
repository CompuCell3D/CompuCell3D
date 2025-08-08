# CompuCell3D Simulation Methods Documentation

This document explains the two different methods for running CompuCell3D simulations using the Jupyter Specification UI.

## Overview

The `SortingExample.ipynb` notebook demonstrates two approaches for running simulations:

1. **Method 1: Easy Way** - Using `ui.run_and_visualize()`
2. **Method 2: Manual Setup** - Getting specs from UI and setting up simulation manually

Both methods achieve the same result but offer different levels of control and complexity.

## Method 1: Easy Way - `ui.run_and_visualize()`

### Description
This is the simplest approach that handles everything internally with a single method call.

### Code Example
```python
# Method 1 - Easy Way: Using ui.run_and_visualize()
ui.run_and_visualize()
```

### What It Does Internally
1. **Validates Configuration**: Checks all UI parameters for validity
2. **Generates Specifications**: Converts UI settings to CC3D specification objects
3. **Initializes Service**: Creates and configures CC3DSimService
4. **Registers Specs**: Automatically registers all specifications
5. **Starts Simulation**: Runs, initializes, and starts the simulation
6. **Creates Visualization**: Displays interactive visualization widgets
7. **Shows Controls**: Displays run/pause button for simulation control

### Advantages
- **Single Method Call**: Everything handled in one line
- **Built-in Validation**: Comprehensive error checking
- **Automatic Widget Creation**: Visualization and controls created automatically
- **Error Handling**: Graceful fallbacks if visualization fails
- **Status Messages**: Detailed progress feedback
- **User-Friendly**: Minimal setup required

### Use Cases
- Quick prototyping and testing
- Educational demonstrations
- When you want maximum convenience
- When you don't need fine-grained control

### Output
- Validation status messages
- Simulation initialization progress
- Interactive visualization widget
- Run/pause control button
- Error messages if validation fails

## Method 2: Manual Setup - Getting Specs and Setting Up Manually

### Description
This approach gives you full control over the simulation setup process by manually handling each step.

### Code Example
```python
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
# Method 2
# Get specifications from the UI
specs = ui.specs

# Create simulation service
cc3d_sim = CC3DSimService()

# Register specifications manually
cc3d_sim.register_specs(specs)

# Set up simulation step by step
cc3d_sim.run()
cc3d_sim.init()
cc3d_sim.start()

# Create visualization manually
vis_widget = cc3d_sim.visualize().show()

# Display run button manually
display(cc3d_sim.jupyter_run_button())
```

### What Each Step Does
1. **`specs = ui.specs`**: Extracts specification objects from UI configuration
2. **`CC3DSimService()`**: Creates new simulation service instance
3. **`register_specs(specs)`**: Registers all specifications with the service
4. **`run()`**: Compiles and prepares the simulation
5. **`init()`**: Initializes the simulation environment
6. **`start()`**: Starts the simulation execution
7. **`visualize().show()`**: Creates and displays visualization widget
8. **`jupyter_run_button()`**: Creates run/pause control button

### Advantages
- **Full Control**: Complete control over each step
- **Customization**: Can modify any step or add custom logic
- **Debugging**: Easier to identify where issues occur
- **Flexibility**: Can integrate with custom workflows
- **Learning**: Better understanding of the underlying process

### Use Cases
- Advanced users who need custom control
- Integration with existing workflows
- Debugging simulation issues
- Educational purposes (understanding the process)
- When you need to modify the setup process

### Output
- Manual control over all outputs
- Can customize visualization and controls
- Direct access to simulation service object

## Key Differences

| Aspect | Method 1 (Easy) | Method 2 (Manual) |
|--------|------------------|-------------------|
| **Complexity** | Single method call | Multiple steps |
| **Control** | Limited | Full control |
| **Customization** | None | Complete |
| **Debugging** | Difficult | Easy |
| **Learning Curve** | Low | High |
| **Error Handling** | Built-in | Manual |
| **Code Length** | 1 line | 8+ lines |

## When to Use Each Method

### Use Method 1 (`ui.run_and_visualize()`) when:
- You want quick results
- You're prototyping or testing
- You're new to CompuCell3D
- You don't need custom modifications
- You want maximum convenience

### Use Method 2 (Manual Setup) when:
- You need custom control over the process
- You're debugging simulation issues
- You want to understand the underlying process
- You need to integrate with existing code
- You want to modify the setup workflow

## Technical Details

### Internal Process of Method 1
```python
def run_and_visualize(self):
    # 1. Validate configuration
    is_valid, errors = self.validate_configuration()

    # 2. Generate specifications
    specs = self.specs

    # 3. Create and configure service
    self.cc3d_sim = CC3DSimService()
    self.cc3d_sim.register_specs(specs)

    # 4. Start simulation
    self.cc3d_sim.run()
    self.cc3d_sim.init()
    self.cc3d_sim.start()

    # 5. Create visualization
    viewer = self.cc3d_sim.visualize().show()

    # 6. Display controls
    run_button = self.cc3d_sim.jupyter_run_button()
    display(run_button)
```

### What `ui.specs` Returns
The `ui.specs` property returns a list of specification objects that can be used with CC3DSimService:

```python
specs = [
    Metadata(...),           # Global simulation settings
    PottsCore(...),          # Lattice and simulation parameters
    CellTypePlugin(...),     # Cell type definitions
    VolumePlugin(...),       # Volume constraints
    ContactPlugin(...),      # Contact energies
    BlobInitializer(...),    # Initial cell distribution
    # ... other plugins as configured
]
```

## Troubleshooting

### Common Issues with Method 1
- **Configuration validation fails**: Check UI parameters
- **Visualization not displayed**: Check CC3D installation
- **Run button not available**: May need manual controls

### Common Issues with Method 2
- **Import errors**: Ensure CC3DSimService is available
- **Specification errors**: Check that specs are properly generated
- **Visualization errors**: Verify CC3D graphics support

## Best Practices

1. **Start with Method 1** for initial testing and prototyping
2. **Use Method 2** when you need more control or are debugging
3. **Always validate configurations** before running simulations
4. **Check error messages** for guidance on fixing issues
5. **Test with simple configurations** before complex setups

## Conclusion

Both methods provide the same end result - a running CompuCell3D simulation with visualization and controls. Choose the method that best fits your needs:

- **Method 1** for convenience and quick results
- **Method 2** for control and customization

The choice depends on your experience level, requirements, and whether you need custom modifications to the simulation setup process.