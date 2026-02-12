from poseblend.schema.run_data import RunData


def run_poseblend(config_path: str, scene_path: str, blender_object_data_path: str) -> None:
    run_data = RunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)