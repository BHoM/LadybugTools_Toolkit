from ladybugtools_toolkit.external_comfort.material import (
    EnergyMaterial,
    EnergyMaterialVegetation,
    Materials,
    OpaqueMaterial,
    OpaqueVegetationMaterial,
    material_from_dict,
)

from .. import BASE_IDENTIFIER

MATERIALS = [
    OpaqueMaterial(BASE_IDENTIFIER, "source", 1, 1, 100, 100),
    OpaqueVegetationMaterial(BASE_IDENTIFIER, "source", 1, 1, 100, 100),
]


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""
    for obj in MATERIALS:
        obj_dict = obj.to_dict()
        assert "_t" in obj_dict.keys()


def test_to_json():
    """Test whether an object can be converted to a json string."""
    for obj in MATERIALS:
        obj_json = obj.to_json()
        assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    for obj in MATERIALS:
        new_obj = type(obj).from_dict(obj.to_dict())
        assert isinstance(new_obj, type(obj))


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    for obj in MATERIALS:
        new_obj = type(obj).from_json(obj.to_json())
        assert isinstance(new_obj, type(obj))


def test_to_lbt():
    """_"""
    for obj in MATERIALS:
        assert isinstance(obj.to_lbt(), (EnergyMaterial, EnergyMaterialVegetation))


def test_from_lbt():
    """_"""
    for obj in MATERIALS:
        assert isinstance(
            type(obj).from_lbt(obj.to_lbt()), (OpaqueMaterial, OpaqueVegetationMaterial)
        )


def test_material_from_dict():
    """_"""
    for obj in MATERIALS:
        assert isinstance(
            material_from_dict(obj.to_dict()),
            (OpaqueMaterial, OpaqueVegetationMaterial),
        )


def test_materials():
    """_"""
    for material in Materials:
        assert isinstance(material.value, (OpaqueMaterial, OpaqueVegetationMaterial))
