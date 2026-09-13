# Compiled Mitosis demos

These demos cover single-cell random division, cluster random division, and periodic-boundary cluster division with an orientation vector. Each example uses `ctx->mitosis.mitosis()`, retrieves parent and child cells, and calls `cloneAttributes`.

`cloneAttributes` copies the cell attributes and plugin state currently represented by the C ABI: AdhesionFlex, PolarizationVector, CellOrientation, LengthConstraint, and Chemotaxis. ContactOrientation, ContactLocalProduct, Polarization23, ConnectivityGlobal/LocalFlex, SBML and dictionary state, MaBoSS state, and FocalPointPlasticity links remain outside this ABI by design.
