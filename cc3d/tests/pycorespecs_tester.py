"""Tests all xml importing in cc3d.core.PyCoreSpecs"""

from os import listdir, walk
from os.path import abspath, dirname, join, splitext

from cc3d.core.PyCoreSpecs import from_file, SpecValueError, _SpecValueErrorContextBlock


tests_dir = join(dirname(dirname(dirname(abspath(__file__)))), "CompuCell3D", "core", "Demos")
"""Directory containing .cc3d files against which to test xml importing"""

patterns_ignore = [
    r"ArticlesPublishedDemos\VascularizedTumor3D_PlosOne2009",  # Redundant declaration in CellType Plugin
    r"BookChapterDemos_ComputationalMethodsInCellBiology\Angiogenesis",  # No XML
    r"BookChapterDemos_ComputationalMethodsInCellBiology\DeltaNotch",  # No XML
    r"BookChapterDemos_ComputationalMethodsInCellBiology\VascularTumor",  # Redundant declaration in Chemotaxis Plugin
    r"BookChapterDemos_ComputationalMethodsInCellBiology\VascularTumor_legacy_implementation",  # Redundant declaration in Chemotaxis Plugin
    r"CompuCellPythonTutorial\CellInitializer",  # Missing XML
    r"SimulationSettings\ParallelCC3DExamples\diffusion-3D",  # No XML
    r"SteppableDemos\DiffusionSolverFE\diffusion_3D_scale_wall"  # Incorrect id declaration in CellType Plugin
]
"""List of patterns to ignore during testing"""
patterns_ignore = [join(tests_dir, x) for x in patterns_ignore]
for f in listdir(join(tests_dir, "CompuCellPythonTutorial", "PythonOnlySimulationsExamples")):
    patterns_ignore.append(join(tests_dir, "CompuCellPythonTutorial", "PythonOnlySimulationsExamples", f))  # No XML


def test_import():
    """
    Test PyCoreSpecs module xml import against all demos

    :return: None
    """
    for dirpath, dirnames, filenames in walk(tests_dir):
        for fn in filenames:
            basename, ext = splitext(fn)
            if ext == ".xml":
                cc3d_filename = join(dirpath, fn)

                if dirname(dirname(cc3d_filename)) in patterns_ignore:
                    continue

            elif ext == ".cc3d":
                cc3d_filename = join(dirpath, fn)

                if dirname(cc3d_filename) in patterns_ignore:
                    continue

            else:
                continue

            print("Testing::", cc3d_filename)
            from_file(cc3d_filename)


def _test_validation_dependencies(*dep_obj, target) -> (bool, str):
    """
    Performs basic checking of dependencies on a target object

    :param dep_obj: dependency instances
    :param target: target instance
    :return: True with SpecValueError message if passed, False with nothing if failed
    :rtype: (bool, str)
    """
    try:
        target.validate()
        return False, ""
    except SpecValueError as e:
        try:
            target.validate(*dep_obj)
            return True, str(e) + ", " + str(e.names)
        except SpecValueError:
            return False, ""


def test_validation_adhesion_flex() -> (str, bool):
    """
    Tests validation defined for AdhesionFlexPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, AdhesionFlexPluginSpecs

    name = AdhesionFlexPluginSpecs.registered_name

    adhesion_specs = AdhesionFlexPluginSpecs(neighbor_order=2)
    specs = [PottsCoreSpecs(), CellTypePluginSpecs("T1"), adhesion_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in adhesion_specs.depends_on], target=adhesion_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    adhesion_specs.density_new(molecule="M1", cell_type="T1", density=1.0)

    # Validation error: T2 not defined
    param = adhesion_specs.density_new(molecule="M1", cell_type="T2", density=1.0)

    # Validation error: T3 not defined
    param.cell_type = "T3"

    try:
        param.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        param.cell_type = "T1"
    try:
        adhesion_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_blob_initializer() -> (str, bool):
    """
    Tests validation defined for BlobInitializer

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, BlobInitializerSpecs
    from cc3d.cpp.CompuCell import Point3D

    name = BlobInitializerSpecs.registered_name

    dim_x, dim_y, dim_z = 2, 2, 2

    blob_init_specs = BlobInitializerSpecs()
    specs = [PottsCoreSpecs(dim_x=dim_x, dim_y=dim_y, dim_z=dim_z), CellTypePluginSpecs("T1"), blob_init_specs]
    reg = blob_init_specs.region_new(radius=1, center=Point3D(1, 1, 1), cell_types=["T1"])

    # Validate dependencies
    res, msg = _test_validation_dependencies(*specs, target=blob_init_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation errors: outside of boundary
    for direction, offset in {"min": -1, "max": 1}.items():
        for comp in ["x", "y", "z"]:
            setattr(reg.center, comp, getattr(reg.center, comp) + offset)
            try:
                reg.validate(*specs)
                return name, False
            except SpecValueError as e:
                print(name, str(e), str(e.names))
                try:
                    blob_init_specs.validate(*specs)
                    return name, False
                except SpecValueError as e:
                    print(name, str(e), str(e.names))
                    setattr(reg.center, comp, getattr(reg.center, comp) - offset)

    # Validation error: undefined cell type
    reg = blob_init_specs.region_new(radius=1, center=Point3D(1, 1, 1), cell_types=["T4"])
    try:
        reg.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        try:
            blob_init_specs.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            pass

    return name, True


def test_validation_chemotaxis() -> (str, bool):
    """
    Tests validation defined for ChemotaxisPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, DiffusionSolverFESpecs, ChemotaxisPluginSpecs

    name = ChemotaxisPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs]

    # Setup: DiffusionSolverFE
    f1_solver_specs = DiffusionSolverFESpecs()
    f1_solver_specs.field_new("F1")
    specs.append(f1_solver_specs)

    chemotaxis_specs = ChemotaxisPluginSpecs()
    specs.append(chemotaxis_specs)

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in chemotaxis_specs.depends_on], target=chemotaxis_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: field name not registered
    cs = chemotaxis_specs.param_new("F2", "DiffusionSolverFE")
    try:
        chemotaxis_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1_solver_specs.field_new("F2")
        try:
            chemotaxis_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: solver not registered
    try:
        chemotaxis_specs.validate(*[s for s in specs if not isinstance(s, DiffusionSolverFESpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        try:
            chemotaxis_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: cell type name not registered
    cs.param_new("T1")
    try:
        chemotaxis_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T1")
        try:
            chemotaxis_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_connectivity_global() -> (str, bool):
    """
    Tests validation defined for ContactPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ConnectivityGlobalPluginSpecs

    name = ConnectivityGlobalPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    connectivity_specs = ConnectivityGlobalPluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, connectivity_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in connectivity_specs.depends_on], target=connectivity_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: cell type name not registered
    connectivity_specs.cell_type_append("T1")
    try:
        connectivity_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T1")
        try:
            connectivity_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_contact() -> (str, bool):
    """
    Tests validation defined for ContactPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ContactPluginSpecs

    name = ContactPluginSpecs.registered_name

    contact_specs = ContactPluginSpecs(neighbor_order=2)
    specs = [PottsCoreSpecs(), CellTypePluginSpecs("T1"), contact_specs]
    param1 = contact_specs.param_new(type_1="Medium", type_2="T1", energy=1.0)

    # Validate dependencies
    res, msg = _test_validation_dependencies(*specs, target=contact_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: T2 not defined
    param1.type_1 = "T2"
    try:
        param1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        param1.type_1 = "Medium"
        param1.type_2 = "T2"
        try:
            param1.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            param1.type_2 = "T1"

    # Validation error: T3 not defined
    contact_specs.param_new(type_1="Medium", type_2="T3", energy=1.0)
    try:
        contact_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_contact_internal() -> (str, bool):
    """
    Tests validation defined for ContactInternalPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ContactInternalPluginSpecs

    name = ContactInternalPluginSpecs.registered_name

    contact_specs = ContactInternalPluginSpecs(neighbor_order=2)
    specs = [PottsCoreSpecs(), CellTypePluginSpecs("T1"), contact_specs]
    param1 = contact_specs.param_new(type_1="Medium", type_2="T1", energy=1.0)

    # Validate dependencies
    res, msg = _test_validation_dependencies(*specs, target=contact_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: T2 not defined
    param1.type_1 = "T2"
    try:
        param1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        param1.type_1 = "Medium"
        param1.type_2 = "T2"
        try:
            param1.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            param1.type_2 = "T1"

    # Validation error: T3 not defined
    contact_specs.param_new(type_1="Medium", type_2="T3", energy=1.0)
    try:
        contact_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_contact_local_flex() -> (str, bool):
    """
    Tests validation defined for ContactLocalFlexPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ContactLocalFlexPluginSpecs

    name = ContactLocalFlexPluginSpecs.registered_name

    contact_specs = ContactLocalFlexPluginSpecs(neighbor_order=2)
    specs = [PottsCoreSpecs(), CellTypePluginSpecs("T1"), contact_specs]
    param1 = contact_specs.param_new(type_1="Medium", type_2="T1", energy=1.0)

    # Validate dependencies
    res, msg = _test_validation_dependencies(*specs, target=contact_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: T2 not defined
    param1.type_1 = "T2"
    try:
        param1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        param1.type_1 = "Medium"
        param1.type_2 = "T2"
        try:
            param1.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            param1.type_2 = "T1"

    # Validation error: T3 not defined
    contact_specs.param_new(type_1="Medium", type_2="T3", energy=1.0)
    try:
        contact_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_curvature() -> (str, bool):
    """
    Tests validation defined for CurvaturePluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, CurvaturePluginSpecs

    name = CurvaturePluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    curvature_specs = CurvaturePluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, curvature_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in curvature_specs.depends_on], target=curvature_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: cell type name not registered
    p = curvature_specs.param_type_new("T1", 1, 1)
    try:
        curvature_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T1")
        try:
            curvature_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    p.cell_type = "T2"
    try:
        curvature_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        p.cell_type = "T1"

    pt = curvature_specs.param_internal_new("T1", "T2", 1, 1)
    try:
        curvature_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            curvature_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    pt.type1 = "T3"
    try:
        curvature_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            curvature_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    pt.type2 = "T4"
    try:
        curvature_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T4")
        try:
            curvature_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_diffusion_solver_fe() -> (str, bool):
    """
    Tests validation defined for DiffusionSolverFESpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, DiffusionSolverFESpecs
    from cc3d.core.PyCoreSpecs import ReactionDiffusionSolverFESpecs

    cell_type_specs = CellTypePluginSpecs("T1", "T2")
    f1_solver_specs = DiffusionSolverFESpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, f1_solver_specs]

    name = f1_solver_specs.registered_name

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in f1_solver_specs.depends_on], target=f1_solver_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    f1 = f1_solver_specs.field_new("F1")

    # Validation error: DiffusionData requires a solver
    try:
        f1.diff_data.validate(*[s for s in specs if not isinstance(s, DiffusionSolverFESpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: SecretionDataSpecs requires a solver
    try:
        f1.spec_dict["secr_data"].validate(*[s for s in specs if not isinstance(s, DiffusionSolverFESpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: Unrecognized field name in DiffusionData
    try:
        f1.diff_data.field_name = "F2"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.diff_data.field_name = "F1"

    # Validation error: Unrecognized cell type names in DiffusionData
    try:
        f1.diff_data.diff_types["T3"] = 0.0
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            f1_solver_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    f1.diff_data.decay_types["T4"] = 0.0
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T4")
        try:
            f1_solver_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: Unrecognized cell type names in SecretionData
    p = f1.secretion_data_new("T1", 0.0, contact_type="T2")
    try:
        p.cell_type = "T5"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        p.cell_type = "T1"
        p.contact_type = "T5"
        try:
            f1_solver_specs.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            p.contact_type = "T2"
            pass

    # Validation error: unrecognized field name
    f1.field_name = "F2"
    try:
        f1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.field_name = "F1"

    # Validation error: non-unique field name
    f2_solver_specs = ReactionDiffusionSolverFESpecs()
    f2_solver_specs.field_new("F1")
    specs.append(f2_solver_specs)
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_external_potential() -> (str, bool):
    """
    Tests validation defined for ExternalPotentialPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ExternalPotentialPluginSpecs

    name = ExternalPotentialPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    ext_pot_specs = ExternalPotentialPluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, ext_pot_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in ext_pot_specs.depends_on], target=ext_pot_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: cell type name not registered
    p = ext_pot_specs.param_new("T1")
    try:
        ext_pot_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T1")
        try:
            ext_pot_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    p.cell_type = "T2"
    try:
        ext_pot_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            ext_pot_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_focal_point_plasticity() -> (str, bool):
    """
    Tests validation defined for FocalPointPlasticityPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, FocalPointPlasticityPluginSpecs

    name = FocalPointPlasticityPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs("T1")
    fpp_specs = FocalPointPlasticityPluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, fpp_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in fpp_specs.depends_on], target=fpp_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: cell type name not registered
    p = fpp_specs.param_new("T1", "T1", lambda_fpp=1, activation_energy=1, target_distance=1, max_distance=1)
    try:
        p.type1 = "T2"
        fpp_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            fpp_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    try:
        p.type2 = "T3"
        fpp_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            fpp_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    fpp_specs.param_new("T1", "T4", lambda_fpp=1, activation_energy=1, target_distance=1, max_distance=1)
    try:
        fpp_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T4")
        try:
            fpp_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_kernel_diffusion_solver() -> (str, bool):
    """
    Tests validation defined for KernelDiffusionSolverSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, KernelDiffusionSolverSpecs
    from cc3d.core.PyCoreSpecs import DiffusionSolverFESpecs

    cell_type_specs = CellTypePluginSpecs("T1", "T2")
    f1_solver_specs = KernelDiffusionSolverSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, f1_solver_specs]

    name = f1_solver_specs.registered_name

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in f1_solver_specs.depends_on], target=f1_solver_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    f1 = f1_solver_specs.field_new("F1")

    # Validation error: DiffusionData requires a solver
    try:
        f1.diff_data.validate(*[s for s in specs if not isinstance(s, KernelDiffusionSolverSpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: SecretionDataSpecs requires a solver
    try:
        f1.spec_dict["secr_data"].validate(*[s for s in specs if not isinstance(s, KernelDiffusionSolverSpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: Unrecognized field name in DiffusionData
    try:
        f1.diff_data.field_name = "F2"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.diff_data.field_name = "F1"

    # Validation error: Unrecognized cell type names in SecretionData
    p = f1.secretion_data_new("T1", 0.0, contact_type="T2")
    try:
        p.cell_type = "T5"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        p.cell_type = "T1"
        p.contact_type = "T5"
        try:
            f1_solver_specs.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            p.contact_type = "T2"
            pass

    # Validation error: unrecognized field name
    f1.field_name = "F2"
    try:
        f1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.field_name = "F1"

    # Validation error: non-unique field name
    f2_solver_specs = DiffusionSolverFESpecs()
    f2_solver_specs.field_new("F1")
    specs.append(f2_solver_specs)
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_length_constraint() -> (str, bool):
    """
    Tests validation defined for LengthConstraintPluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, LengthConstraintPluginSpecs

    name = LengthConstraintPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs("T1")
    length_specs = LengthConstraintPluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, length_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in length_specs.depends_on], target=length_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    param = length_specs.param_new("T1", target_length=1.0, lambda_length=1.0)

    # Validation error: undefined cell type
    param.cell_type = "T2"
    try:
        param.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            length_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: undefined cell type
    length_specs.param_new("T3", target_length=1.0, lambda_length=1.0)
    try:
        length_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            length_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_pde_boundary_condition() -> (str, bool):
    """
    Tests validation defined for PDEBoundaryConditionsSpec

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, PDEBoundaryConditionsSpec, BOUNDARYTYPESPOTTS, BOUNDARYTYPESPDE

    name = "PDEBoundaryConditions"

    potts_specs = PottsCoreSpecs()
    bcs = PDEBoundaryConditionsSpec()
    specs = [potts_specs, bcs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in bcs.depends_on], target=bcs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    directions = ["min", "max"]
    pde_d, pde_n, pde_p = BOUNDARYTYPESPDE
    potts_n, potts_p = BOUNDARYTYPESPOTTS
    for c in ["x", "y", "z"]:
        potts_attr = f"boundary_{c}"

        # Validation error: Periodic PDE + No Flux Potts
        setattr(potts_specs, potts_attr, potts_n)
        for d in directions:
            test = False
            bc_attr = f"{c}_{d}_type"
            setattr(bcs, bc_attr, pde_d)
            try:
                bcs.validate(*specs)
                setattr(bcs, bc_attr, pde_n)
                try:
                    bcs.validate(*specs)
                    setattr(bcs, bc_attr, pde_p)
                    try:
                        bcs.validate(*specs)
                    except SpecValueError as e:
                        print(name, str(e), str(e.names))
                        test = True
                except SpecValueError as e:
                    print(name, str(e), str(e.names))
                    pass
            except SpecValueError as e:
                print(name, str(e), str(e.names))
                pass
            if not test:
                return name, False

        # Validation error: Constant value/flux PDE + Periodic Potts
        setattr(potts_specs, potts_attr, potts_p)
        for d in directions:
            test = False
            bc_attr = f"{c}_{d}_type"
            setattr(bcs, bc_attr, pde_d)
            try:
                bcs.validate(*specs)
            except SpecValueError as e:
                print(name, str(e), str(e.names))
                setattr(bcs, bc_attr, pde_n)
                try:
                    bcs.validate(*specs)
                except SpecValueError as e:
                    print(name, str(e), str(e.names))
                    setattr(bcs, bc_attr, pde_p)
                    try:
                        bcs.validate(*specs)
                        test = True
                    except SpecValueError as e:
                        print(name, str(e), str(e.names))
                        pass
            if not test:
                return name, False

    return name, True


def test_validation_potts() -> (str, bool):
    """
    Tests validation defined for PottsCoreSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import CellTypePluginSpecs, PottsCoreSpecs

    name = PottsCoreSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    potts_specs = PottsCoreSpecs()
    specs = [cell_type_specs, potts_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in potts_specs.depends_on], target=potts_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: undefined cell type
    potts_specs.fluctuation_amplitude_type("T1", 0.0)

    try:
        potts_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T1")
        try:
            potts_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_reaction_diffusion_solver_fe() -> (str, bool):
    """
    Tests validation defined for ReactionDiffusionSolverFESpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, ReactionDiffusionSolverFESpecs
    from cc3d.core.PyCoreSpecs import DiffusionSolverFESpecs

    cell_type_specs = CellTypePluginSpecs("T1", "T2")
    f1_solver_specs = ReactionDiffusionSolverFESpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, f1_solver_specs]

    name = f1_solver_specs.registered_name

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in f1_solver_specs.depends_on], target=f1_solver_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    f1 = f1_solver_specs.field_new("F1")

    # Validation error: DiffusionData requires a solver
    try:
        f1.diff_data.validate(*[s for s in specs if not isinstance(s, ReactionDiffusionSolverFESpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: SecretionDataSpecs requires a solver
    try:
        f1.spec_dict["secr_data"].validate(*[s for s in specs if not isinstance(s, ReactionDiffusionSolverFESpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: Unrecognized field name in DiffusionData
    try:
        f1.diff_data.field_name = "F2"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.diff_data.field_name = "F1"

    # Validation error: Unrecognized cell type names in DiffusionData
    try:
        f1.diff_data.diff_types["T3"] = 0.0
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            f1_solver_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    f1.diff_data.decay_types["T4"] = 0.0
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T4")
        try:
            f1_solver_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: Unrecognized cell type names in SecretionData
    p = f1.secretion_data_new("T1", 0.0, contact_type="T2")
    try:
        p.cell_type = "T5"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        p.cell_type = "T1"
        p.contact_type = "T5"
        try:
            f1_solver_specs.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            p.contact_type = "T2"
            pass

    # Validation error: unrecognized field name
    f1.field_name = "F2"
    try:
        f1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.field_name = "F1"

    # Validation error: non-unique field name
    f2_solver_specs = DiffusionSolverFESpecs()
    f2_solver_specs.field_new("F1")
    specs.append(f2_solver_specs)
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_secretion() -> (str, bool):
    """
    Tests validation defined for SteadyStateDiffusionSolverSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, DiffusionSolverFESpecs, SecretionPluginSpecs
    from cc3d.core.PyCoreSpecs import BoundaryPixelTrackerPluginSpecs, PixelTrackerPluginSpecs

    name = SecretionPluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs()
    secretion_specs = SecretionPluginSpecs()
    pde_solver = DiffusionSolverFESpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, secretion_specs, pde_solver,
             PixelTrackerPluginSpecs(), BoundaryPixelTrackerPluginSpecs()]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in secretion_specs.depends_on], target=secretion_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    # Validation error: disabled pixel tracker
    try:
        secretion_specs.pixel_tracker = False
        secretion_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        try:
            secretion_specs.validate(*[s for s in specs if not isinstance(s, PixelTrackerPluginSpecs)])
            secretion_specs.pixel_tracker = True
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: disabled boundary pixel tracker
    try:
        secretion_specs.boundary_pixel_tracker = False
        secretion_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        try:
            secretion_specs.validate(*[s for s in specs if not isinstance(s, BoundaryPixelTrackerPluginSpecs)])
            secretion_specs.boundary_pixel_tracker = True
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: field name not registered
    fs = secretion_specs.field_new(field_name="F1")
    try:
        secretion_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pde_solver.field_new("F1")
        try:
            secretion_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    fs.field_name = "F2"
    try:
        secretion_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pde_solver.field_new("F2")
        try:
            secretion_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_steady_state_diffusion_solver() -> (str, bool):
    """
    Tests validation defined for SteadyStateDiffusionSolverSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, SteadyStateDiffusionSolverSpecs
    from cc3d.core.PyCoreSpecs import DiffusionSolverFESpecs

    cell_type_specs = CellTypePluginSpecs("T1", "T2")
    f1_solver_specs = SteadyStateDiffusionSolverSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, f1_solver_specs]

    name = f1_solver_specs.registered_name

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in f1_solver_specs.depends_on], target=f1_solver_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    f1 = f1_solver_specs.field_new("F1")

    # Validation error: DiffusionData requires a solver
    try:
        f1.diff_data.validate(*[s for s in specs if not isinstance(s, SteadyStateDiffusionSolverSpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: SecretionDataSpecs requires a solver
    try:
        f1.spec_dict["secr_data"].validate(*[s for s in specs if not isinstance(s, SteadyStateDiffusionSolverSpecs)])
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    # Validation error: Unrecognized field name in DiffusionData
    try:
        f1.diff_data.field_name = "F2"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.diff_data.field_name = "F1"

    # Validation error: Unrecognized cell type names in SecretionData
    p = f1.secretion_data_new("T1", 0.0)
    try:
        p.cell_type = "T5"
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        p.cell_type = "T1"

    # Validation error: unrecognized field name
    f1.field_name = "F2"
    try:
        f1.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        f1.field_name = "F1"

    # Validation error: non-unique field name
    f2_solver_specs = DiffusionSolverFESpecs()
    f2_solver_specs.field_new("F1")
    specs.append(f2_solver_specs)
    try:
        f1_solver_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        pass

    return name, True


def test_validation_surface() -> (str, bool):
    """
    Tests validation defined for SurfacePluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, SurfacePluginSpecs

    name = SurfacePluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs("T1")
    surface_specs = SurfacePluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, surface_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in surface_specs.depends_on], target=surface_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    param = surface_specs.param_new("T1", target_surface=1.0, lambda_surface=1.0)

    # Validation error: undefined cell type
    param.cell_type = "T2"
    try:
        param.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            surface_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: undefined cell type
    surface_specs.param_new("T3", target_surface=1.0, lambda_surface=1.0)
    try:
        surface_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            surface_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation_uniform_initializer() -> (str, bool):
    """
    Tests validation defined for UniformInitializerSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import CellTypePluginSpecs, PottsCoreSpecs, UniformInitializerSpecs
    from cc3d.cpp.CompuCell import Point3D

    name = UniformInitializerSpecs.registered_name

    ct = ["T1", "T2", "T3"]
    dim_x, dim_y, dim_z = 1, 2, 3
    unif_init_specs = UniformInitializerSpecs()
    specs = [PottsCoreSpecs(dim_x=dim_x, dim_y=dim_y), CellTypePluginSpecs(*ct), unif_init_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in unif_init_specs.depends_on], target=unif_init_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    reg = unif_init_specs.region_new(pt_min=Point3D(0, 0, 0), pt_max=Point3D(dim_x, dim_y, dim_z), cell_types=ct)

    # Validation errors: outside of boundary
    for direction, offset in {"min": -1, "max": 1}.items():
        for comp in ["x", "y", "z"]:
            attr = f"pt_{direction}"
            setattr(getattr(reg, attr), comp, getattr(getattr(reg, attr), comp) + offset)
            try:
                reg.validate(*specs)
                return name, False
            except SpecValueError as e:
                print(name, str(e), str(e.names))
                try:
                    unif_init_specs.validate(*specs)
                    return name, False
                except SpecValueError as e:
                    print(name, str(e), str(e.names))
                    setattr(getattr(reg, attr), comp, getattr(getattr(reg, attr), comp) - offset)

    # Validation error: undefined cell type
    reg = unif_init_specs.region_new(pt_min=Point3D(0, 0, 0), pt_max=Point3D(dim_x, dim_y, dim_z), cell_types=["T4"])
    try:
        reg.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        try:
            unif_init_specs.validate(*specs)
            return name, False
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            pass

    return name, True


def test_validation_volume() -> (str, bool):
    """
    Tests validation defined for VolumePluginSpecs

    :return: registered name and test result
    :rtype: (str, bool)
    """
    from cc3d.core.PyCoreSpecs import PottsCoreSpecs, CellTypePluginSpecs, VolumePluginSpecs

    name = VolumePluginSpecs.registered_name

    cell_type_specs = CellTypePluginSpecs("T1")
    volume_specs = VolumePluginSpecs()
    specs = [PottsCoreSpecs(), cell_type_specs, volume_specs]

    # Validate dependencies
    res, msg = _test_validation_dependencies(*[d() for d in volume_specs.depends_on], target=volume_specs)
    if res:
        print(name), print(msg)
    else:
        return name, False

    param = volume_specs.param_new("T1", target_volume=1.0, lambda_volume=1.0)

    # Validation error: undefined cell type
    param.cell_type = "T2"
    try:
        param.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T2")
        try:
            volume_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    # Validation error: undefined cell type
    volume_specs.param_new("T3", target_volume=1.0, lambda_volume=1.0)
    try:
        volume_specs.validate(*specs)
        return name, False
    except SpecValueError as e:
        print(name, str(e), str(e.names))
        cell_type_specs.cell_type_append("T3")
        try:
            volume_specs.validate(*specs)
        except SpecValueError as e:
            print(name, str(e), str(e.names))
            return name, False

    return name, True


def test_validation():
    """
    Tests validation

    :return: None
    """
    print("Testing validation")
    validation_tests = [
        test_validation_adhesion_flex,
        test_validation_blob_initializer,
        test_validation_chemotaxis,
        test_validation_connectivity_global,
        test_validation_contact,
        test_validation_contact_internal,
        test_validation_contact_local_flex,
        test_validation_curvature,
        test_validation_diffusion_solver_fe,
        test_validation_external_potential,
        test_validation_focal_point_plasticity,
        test_validation_kernel_diffusion_solver,
        test_validation_length_constraint,
        test_validation_pde_boundary_condition,
        test_validation_potts,
        test_validation_reaction_diffusion_solver_fe,
        test_validation_secretion,
        test_validation_steady_state_diffusion_solver,
        test_validation_surface,
        test_validation_uniform_initializer,
        test_validation_volume
    ]

    with _SpecValueErrorContextBlock() as err_ctx:
        for vt in validation_tests:
            with err_ctx.ctx:
                name, result = vt()
                if result:
                    print("Validation passed:", name)
                else:
                    raise SpecValueError("Validation failed: " + name)


def test_serial() -> None:
    """
    Tests whether all main specs can be serialized

    :return: None
    """
    from cc3d.core.PyCoreSpecs import PLUGINS, STEPPABLES, INITIALIZERS, MetadataSpecs
    from cc3d.core.PyCoreSpecs import PIFDumperSteppableSpecs, PIFInitializerSteppableSpecs
    import pickle

    ins = {PIFDumperSteppableSpecs: ([], {"pif_name": "dummy", "frequency": 1}),
           PIFInitializerSteppableSpecs: ([], {"pif_name": "dummy"})}

    for spec in PLUGINS + STEPPABLES + INITIALIZERS + [MetadataSpecs]:
        print("Testing serialization:", spec)

        try:
            args, kwargs = ins[spec]
        except KeyError:
            args, kwargs = [], {}

        o = spec(*args, **kwargs)
        dumpo = pickle.dumps(o)
        loado = pickle.loads(dumpo)
        print(o, loado, dumpo)


def main():
    """
    Performs tests

    :return: None
    """
    with _SpecValueErrorContextBlock() as err_ctx:
        err_ctx.ctx(test_import)
        err_ctx.ctx(test_validation)
        err_ctx.ctx(test_serial)


if __name__ == "__main__":
    main()
