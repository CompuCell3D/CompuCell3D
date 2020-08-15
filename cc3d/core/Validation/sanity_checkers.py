import re


def validate_cc3d_entity_identifier(entity_identifier: str, entity_type_label:str= ''):
    """
    Checks if entity identifier conforms to CC3D naming standards. Cell types names, fields etc
    have to start from the letter and contain only alphanumeric strings with no spaces or charcters that could be
    interpreted as syntax elements in python . Function does not return anything but raises AttributeError
    exception if it detects inappropriate identifier
    :param entity_identifier: identifier to check
    :param entity_type_label: optional string that helps print out more-targeted error message
    :return:
    """
    alphanumeric_underscore_regex = r'^\w+$'

    if not entity_type_label:
        entity_type_label = 'identifier'

    if entity_identifier.isspace() or not len(entity_identifier.strip()):
        raise AttributeError(f'{entity_type_label} "{entity_identifier}" contains whitespaces')

    if not entity_identifier[0].isalpha():
        raise AttributeError(f'Invalid {entity_type_label} "{entity_identifier}" . {entity_type_label} '
                             f'must start with a letter')

    if not re.match(alphanumeric_underscore_regex, entity_identifier):
        raise AttributeError(f'Invalid character detected in a {entity_type_label} "{entity_identifier}" . '
                             f'{entity_type_label} must consist of alphanumeric characters and (optionally) '
                             f'an underscore')
