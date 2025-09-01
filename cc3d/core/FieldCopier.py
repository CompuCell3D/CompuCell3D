from cc3d import CompuCellSetup
from cc3d.cpp import PlayerPython


def copy_cell_attribute_field_values_to(field_name:str, cell_attribute_name:str):

    pg = CompuCellSetup.persistent_globals
    field_copier = PlayerPython.FieldCopier(pg.simulator)
    available_attributes = field_copier.get_available_attributes()
    if cell_attribute_name not in available_attributes:
        raise ValueError(f"cell_attribute_name={cell_attribute_name} not in available_attributes={available_attributes}")
    field_copier.copy_cell_attribute_field_values_to(field_name, cell_attribute_name)

def copy_legacy_concentration_field(source_field_name:str, destination_field_name):
    pg = CompuCellSetup.persistent_globals
    field_copier = PlayerPython.FieldCopier(pg.simulator)
    field_copier.copy_legacy_concentration_field(source_field_name=source_field_name, destination_field_name=destination_field_name)