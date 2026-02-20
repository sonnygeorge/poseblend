import argparse
import asyncio
import json
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from poseblend.run import run_poseblend

load_dotenv()

# CONFIG_PATH_DEFAULT = "inputs/configs/simple.yaml"
CONFIG_PATH_DEFAULT = "inputs/configs/recommended.yaml"
SCENE_PATH_DEFAULT = "inputs/scenes/dog_throws_laptop_over_sign_to_person.yaml"
# SCENE_PATH_DEFAULT = "inputs/scenes/horse_shows_bird_to_person_violin_nearby.yaml"
SCENE_PATH_DEFAULT = "inputs/scenes/horse_rides_astronaut.yaml"
BLENDER_OBJECT_DATA_PATH_DEFAULT = "inputs/blender_object_data.yaml"
DEFAULT_GUI_PORT = 8420


async def _run_with_gui(args: argparse.Namespace) -> None:
    import uvicorn

    from poseblend.gui.server import create_app
    from poseblend.schema.run_data import RunData

    run_data = RunData.from_input_yaml_paths(args.config, args.scene, args.blender_object_data)
    app, broadcast = create_app(run_data, run_data.run_dir)

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)
    webbrowser.open(f"http://127.0.0.1:{args.port}")

    try:
        await run_poseblend(
            args.config,
            args.scene,
            args.blender_object_data,
            on_update=broadcast,
            run_data=run_data,
        )
        # Keep server alive after pipeline completes until Ctrl+C
        await server_task
    except asyncio.CancelledError:
        pass
    finally:
        server.should_exit = True
        await server_task


async def _view_run(args: argparse.Namespace) -> None:
    import uvicorn

    from poseblend.gui.server import create_app
    from poseblend.schema.run_data import RunData

    view_path = Path(args.view)
    if view_path.is_dir():
        json_path = view_path / "run_data.json"
    else:
        json_path = view_path
        view_path = json_path.parent

    if not json_path.exists():
        msg = f"run_data.json not found at {json_path}"
        raise FileNotFoundError(msg)

    raw = json.loads(json_path.read_text())
    run_data = RunData.model_validate(raw)
    app, _broadcast = create_app(run_data, view_path)

    webbrowser.open(f"http://127.0.0.1:{args.port}")

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch real-time GUI alongside pipeline",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GUI_PORT,
        help=f"GUI server port (default: {DEFAULT_GUI_PORT})",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        help="Path to a run directory or run_data.json to view in the GUI (post-hoc mode)",
    )
    args = parser.parse_args()

    if args.gui and args.view:
        parser.error("--gui and --view are mutually exclusive")

    if args.view:
        asyncio.run(_view_run(args))
    elif args.gui:
        asyncio.run(_run_with_gui(args))
    else:
        asyncio.run(run_poseblend(args.config, args.scene, args.blender_object_data))


if __name__ == "__main__":
    main()
