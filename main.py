import argparse

from dotenv import load_dotenv

from poseblend.run import run_poseblend

load_dotenv()

CONFIG_PATH_DEFAULT = "inputs/configs/simple.yaml"
SCENE_PATH_DEFAULT = "inputs/scenes/person_throws_laptop_over_sign_to_dog.yaml"
BLENDER_OBJECT_DATA_PATH_DEFAULT = "inputs/blender_object_data.yaml"


def main():
    parser = argparse.ArgumentParser(description="PoseBlend image generation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH_DEFAULT,
        help=f"Path to the config YAML file (default: {CONFIG_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=SCENE_PATH_DEFAULT,
        help=f"Path to the scene spec YAML file (default: {SCENE_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--blender-object-data",
        type=str,
        default=BLENDER_OBJECT_DATA_PATH_DEFAULT,
        help=f"Path to the blender object data YAML file (default: {BLENDER_OBJECT_DATA_PATH_DEFAULT})",
    )
    args = parser.parse_args()

    run_poseblend(args.config, args.scene, args.blender_object_data)


if __name__ == "__main__":
    main()
