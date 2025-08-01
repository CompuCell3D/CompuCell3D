"""
Jupyter Widget Styling Module

Provides CSS styling for CompuCell3D Jupyter interface widgets.
Injects styles into notebook environment for consistent appearance.

Why separate file:
- Separation of concerns (styling vs business logic)
- Reusable across notebooks
- Centralized maintenance
- Standard Jupyter CSS injection approach
- **NOTE**: Prevents styling loss on cell re-execution
  When styling is injected inside a class (e.g., SpecificationSetupUI),
  rerunning the cell with `ui = SpecificationSetupUI()` causes the styling
  to disappear due to Jupyter's cell execution model. This separate file
  allows persistent styling that survives cell re-execution.
"""

from IPython.display import display, HTML as IPythonHTML

def inject_jupyter_widget_css():
    """Injects CSS styles into Jupyter notebook for widget styling."""
    display(IPythonHTML("""
    <style>
    /* Round corners for all input boxes */
    .widget-text input,
    .widget-bounded-int-text input,
    .widget-bounded-float-text input,
    .widget-float-text input,
    .widget-int-text input {
        border-radius: 4px !important;
    }

    /* Round corners for dropdown/select inputs */
    .widget-dropdown select,
    .widget-select select {
        border-radius: 4px !important;
    }

    /* Round corners for buttons */
    .widget-button button,
    .jupyter-button {
        border-radius: 4px !important;
    }

    /* Spacing classes for layout containers */
    .vbox-row-spacing {
        margin: 10px 0 !important;
    }

    .hbox-item-spacing {
        margin: 0 15px 0 0 !important;
    }

    .vbox-no-margin {
        margin: 0 !important;
    }

    .hbox-no-margin {
        margin: 0 !important;
    }

    .celltype-item-spacing {
        margin: 0 0 5px 0 !important;
    }

    .small-right-spacing {
        margin: 0 5px 0 0 !important;
    }

    /* Plugin-specific spacing classes */
    .plugin-config-container {
        padding: 0 0 0 0 !important;
    }
    .plugin-compact-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    .plugin-input-spacing {
        margin: 0 15px 0 0 !important;
    }
    .plugin-bottom-spacing {
        margin: 0 0 15px 0 !important;
    }
    .plugin-checkbox-bottom-spacing {
        margin: 0 0 15px 0 !important;
    }
    .button-spacing {
        margin: 10px 0 !important;
    }

    /* Error state styling with rounded corners */
    .error-input input {
        border: 2px solid #f44336 !important;
        background-color: #ffebee !important;
        box-shadow: 0 0 3px rgba(244, 67, 54, 0.3) !important;
        border-radius: 4px !important;
    }

    .error-input input:focus {
        border-color: #d32f2f !important;
        box-shadow: 0 0 5px rgba(244, 67, 54, 0.5) !important;
        border-radius: 4px !important;
    }

    .plugin-top-spacing {
        margin-top: 10px !important;
    }
    </style>
    """))