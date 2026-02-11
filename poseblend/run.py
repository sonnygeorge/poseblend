from poseblend.schema.run_data import PoseBlendRunData


def run_poseblend(config_path: str, scene_path: str, blender_object_data_path: str) -> None:
    run_data = PoseBlendRunData.from_input_yaml_paths(config_path, scene_path, blender_object_data_path)