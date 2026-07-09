# Fixture Layer Draft

The native fixture layer constructs a real `Simulator`/`Potts3D` test harness using in-memory `CC3DXMLElement`
configuration, without parsing external XML files.

Current responsibilities:

- minimal `<Potts>` XML construction
- minimal `<Plugin>` XML construction
- simulator initialization through `initializeCC3D()` and `extraInit()`
- cell creation and field painting using the real cell field
