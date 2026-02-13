import dataclasses

from poseblend.blender.schema import BlenderObjectMetadata as DC_BlenderObjectMetadata
from poseblend.blender.schema import ObjectPlacementParams as DC_ObjectPlacementParams
from poseblend.schema.inputs.blender_objects import BlenderObjectMetadata as Pydantic_BlenderObjectMetadata
from poseblend.schema.lm_outputs import ObjectPlacementParams as Pydantic_ObjectPlacementParams

DATACLASS_PYDANTIC_PAIRS = [
    (DC_BlenderObjectMetadata, Pydantic_BlenderObjectMetadata),
    (DC_ObjectPlacementParams, Pydantic_ObjectPlacementParams),
]


def _dc_field_map(cls: type) -> dict[str, type]:
    return {f.name: f.type for f in dataclasses.fields(cls)}


def _pydantic_field_map(cls: type) -> dict[str, type]:
    return {name: info.annotation for name, info in cls.model_fields.items()}


def test_dataclass_fields_are_subset_of_pydantic_counterpart():
    for dc_cls, pydantic_cls in DATACLASS_PYDANTIC_PAIRS:
        dc_fields = _dc_field_map(dc_cls)
        pydantic_fields = _pydantic_field_map(pydantic_cls)

        missing = set(dc_fields) - set(pydantic_fields)
        assert not missing, (
            f"Dataclass {dc_cls.__name__} has fields not present in pydantic model "
            f"{pydantic_cls.__name__}: {missing}"
        )

        for name, dc_type in dc_fields.items():
            pydantic_type = pydantic_fields[name]
            assert dc_type == pydantic_type, (
                f"Type mismatch for field '{name}' in {dc_cls.__name__}: "
                f"dataclass has {dc_type}, pydantic model has {pydantic_type}"
            )
