from honeybee.model import Model


from python_toolkit.bhom.analytics import analytics


@analytics
def equality(model0: Model, model1: Model, include_identifier: bool = False) -> bool:
    """Check for equality between two models, with regards to their material properties.

    Args:
        model0 (Model):
            A honeybee model.
        model1 (Model):
            A honeybee model.
        include_identifier (bool, optional):
            Include the identifier (name) of the model in the quality check. Defaults to False.

    Returns:
        bool:
            True if models are equal.
    """

    if not isinstance(model0, Model) or not isinstance(model1, Model):
        raise TypeError("Both inputs must be of type Model.")

    if include_identifier:
        if model0.identifier != model1.identifier:
            return False

    # Check ground material properties
    gnd0_material = model0.faces[5].properties.energy.construction.materials[0]
    gnd1_material = model1.faces[5].properties.energy.construction.materials[0]
    gnd_materials_match: bool = str(gnd0_material) == str(gnd1_material)

    # Check shade material properties
    shd0_material = model0.faces[-6].properties.energy.construction.materials[0]
    shd1_material = model1.faces[-6].properties.energy.construction.materials[0]
    shd_materials_match: bool = str(shd0_material) == str(shd1_material)

    return gnd_materials_match and shd_materials_match
