# Contributing Guidelines - CompuCell3D

Thank you for considering contributing to the CompuCell3D project! To maintain a consistent and high-quality codebase, we encourage you to follow these coding standards when submitting your contributions.

## C++ Coding Standards

### General Guidelines
- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
- Use meaningful variable and function names.
- Keep lines under 80 characters whenever possible.
- Indentation should be done with spaces, not tabs.
- Use C++11. We tend to be conservative in terms of compiler choice and strive to ensure that our code compiles with older compilers.

### Class and Function Design
- Classes and functions should have a clear and specific purpose.
- Aim for low coupling and high cohesion.
- Prefer member initialization lists over assignment in the constructor body.
- Rely on Standard Template Library (STL) and resist reinventing functionality that STL already provides.
- Design modules so that they are open for extension but closed for modification (the Open-Closed Principle).

### Memory Management
- Prefer smart pointers (e.g., `std::unique_ptr`, `std::shared_ptr`) over raw pointers - unless you are writing performance-critical module.
- Use RAII principles for resource management.

### Error Handling
- Use exceptions judiciously; prefer error codes or alternatives when appropriate.
- Document error handling strategies in comments.

## Python Coding Standards

### General Guidelines
- Follow the [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/).
- Use meaningful variable and function names.
- Make code self-explanatory. "The proper use of comments is to compensate for our failure to express ourselves in code," as Robert C Martin says. 
- Keep lines under 120 characters whenever possible.
- Indentation should be done with 4 spaces.
- black formatting is encouraged but not requires

### Module and Function Design
- Modules and functions should have a *single* clear and specific purpose.
- Aim for low coupling and high cohesion.

### Documentation
- Include docstrings for all modules, classes, and functions.
- Follow the [NumPy/SciPy Documentation Standards](https://numpydoc.readthedocs.io/en/latest/format.html).

### Version Compatibility
- Code should be compatible with Python 3.x.
- Consider specifying minimum Python version requirements in documentation.

## Pull Request Guidelines

### Before Submitting a Pull Request
- Ensure your code adheres to the coding standards outlined above.
- Run unit tests to verify the changes.
- Update documentation if applicable.

### Pull Request Content
- Add a description of how to test your future so that someone else can test it.
- Add a simulation to the `Demos` folder.
- Reference any related issues or pull requests. Example: "Closes #2".
- If introducing a new feature, include relevant documentation updates.
- Attach screenshots or movies (when relevant) to show your feature in action.

Thank you for your contributions to the CompuCell3D project! Your adherence to these coding standards helps maintain a clean and cohesive codebase.
