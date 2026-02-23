import asyncio
import csv
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from poseblend.models.registry import get_t2i_model
from poseblend.run import run_poseblend
from poseblend.schema.inputs.config import AttemptRangeModelSelectionProbs, PoseBlendConfig
from poseblend.schema.inputs.scene_spec import PoseBlendSceneSpec
from poseblend.utils import download_image, list_grammatically

load_dotenv()

BASE_CONFIG = "inputs/configs/experiment.yaml"
BLENDER_OBJECT_DATA = "inputs/blender_object_data.yaml"
SCENE_PATH_TEMPLATE = "inputs/scenes/{scene_name}.yaml"
MAX_RUNS_PER_SCENE = 18
DEFAULT_EDIT_MODEL = "fal-ai/gemini-25-flash-image/edit"
DEFAULT_T2I_MODEL = "fal-ai/gemini-25-flash-image"
ALL_MODES = ["poseblend", "t2i_simple", "t2i_detailed"]

GENERATION_FIELDS = [
    "scene", "mode", "filepath", "prompt", "poseblend_run_id", "edit_chain_idx",
]
RUN_LOG_FIELDS = ["scene", "run_id", "run_idx", "num_successes", "num_edit_chains"]

SCENES_TO_CONFIG_OVERRIDES: dict[str, dict] = {
    "ball_kicks_athlete_toward_goal": {
        "num_scenes": 2,
        "num_renders": 11,
        "num_edit_chains": 6,
        "contraction_strength": 0.47
    },
    "chandalier_hangs_from_person": {
        "num_scenes": 2,
        "num_renders": 6,
        "num_edit_chains": 3,
        "contraction_strength": 1.1
    },
    "clay_pot_throws_pottery_wheel_at_woman": {
        "num_scenes": 2,
        "num_renders": 12,
        "num_edit_chains": 6,
        "render_resolution_x": 1024,
        "render_resolution_y": 1024,
        "contraction_strength": 0.47,
    },
    "dog_throws_laptop_over_sign_to_person": {
        "num_scenes": 1,
        "num_renders": 14,
        "num_edit_chains": 5,
        "contraction_strength": 0.57
    },
    "flower_flies_over_bee": {
        "num_scenes": 2,
        "num_renders": 7,
        "num_edit_chains": 3,
        "contraction_strength": 0.94
    },
    "horse_rides_astronaut": {
        "num_scenes": 3,
        "num_renders": 7,
        "num_edit_chains": 5,
        "contraction_strength": 0.73
    },
    "horse_shows_bird_to_person_violin_nearby": {
       "num_scenes": 2,
       "num_renders": 9,
       "num_edit_chains": 5,
       "contraction_strength": 0.43,
    },
    "matador_charges_bull": {
        "num_scenes": 1,
        "num_renders": 6,
        "num_edit_chains": 3,
        "contraction_strength": 0.5,
    },
    "mouse_chases_cat": {
      "num_scenes": 1,
      "num_renders": 6,
      "num_edit_chains": 3,
      "contraction_strength": 1.5,
    },
    "player_lobs_hoop_toward_basketball": {
      "num_scenes": 2,
      "num_renders": 8,
      "num_edit_chains": 5,
      "contraction_strength": 0.47,
    },
}


def _build_detailed_t2i_prompt(scene_spec: PoseBlendSceneSpec) -> str:
    all_reqs = [
        req
        for group in scene_spec.get_hydrated_edit_requirements()
        for req in group
    ] + scene_spec.get_hydrated_final_requirements()
    all_reqs = [r[0].lower() + r[1:] for r in all_reqs]
    requirements_str = list_grammatically(all_reqs, enumerate=True)
    requirements_str = requirements_str[0].upper() + requirements_str[1:]
    return (
        f"{scene_spec.scene_as_natural_language} {requirements_str}."
    )


def _load_scene_spec(scene_name: str) -> PoseBlendSceneSpec:
    with open(SCENE_PATH_TEMPLATE.format(scene_name=scene_name)) as f:
        return PoseBlendSceneSpec(**yaml.safe_load(f))


def _build_config(scene_name: str, edit_model: str | None = None) -> PoseBlendConfig:
    with open(BASE_CONFIG) as f:
        config_data: dict = yaml.safe_load(f)
    config_data.update(SCENES_TO_CONFIG_OVERRIDES[scene_name])
    if edit_model is not None:
        config_data["edit_model_selection_schedule"] = [
            AttemptRangeModelSelectionProbs(
                from_attempt=0, to_attempt=None, model_probs={edit_model: 1.0},
            )
        ]
    return PoseBlendConfig(**config_data)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def _run_poseblend_mode(
    output_dir: Path,
    min_successful: int,
    edit_model: str,
    generations: list[dict],
    run_log: list[dict],
) -> None:
    for scene_name in SCENES_TO_CONFIG_OVERRIDES:
        scene_spec = _load_scene_spec(scene_name)
        config = _build_config(scene_name, edit_model=edit_model)
        scene_dir = output_dir / scene_name / "poseblend"
        scene_dir.mkdir(parents=True, exist_ok=True)
        num_successes = 0
        img_idx = 0
        for run_idx in range(MAX_RUNS_PER_SCENE):
            try:
                run_data = await run_poseblend(
                    config=config,
                    scene=scene_spec,
                    blender_object_data=BLENDER_OBJECT_DATA,
                )
            except Exception:
                logger.exception(f"[poseblend][{scene_name}] run {run_idx} failed")
                continue
            run_successes = run_data.num_successful_generations
            run_log.append({
                "scene": scene_name,
                "run_id": run_data.run_id,
                "run_idx": run_idx,
                "num_successes": run_successes,
                "num_edit_chains": len(run_data.edit_chains),
            })
            for chain_idx, ec in enumerate(run_data.edit_chains):
                if ec.gate_decision and ec.gate_decision.is_passing and ec.candidate_final_img_path:
                    dest = scene_dir / f"{img_idx:03d}.png"
                    shutil.copy2(ec.candidate_final_img_path, dest)
                    generations.append({
                        "scene": scene_name,
                        "mode": "poseblend",
                        "filepath": str(dest.relative_to(output_dir)),
                        "prompt": "",
                        "poseblend_run_id": run_data.run_id,
                        "edit_chain_idx": chain_idx,
                    })
                    img_idx += 1
            num_successes += run_successes
            logger.info(
                f"[poseblend][{scene_name}] run {run_idx}: {run_successes} successes "
                f"({num_successes} total)"
            )
            if num_successes >= min_successful:
                break
        else:
            logger.warning(
                f"[poseblend][{scene_name}] only got {num_successes}/{min_successful} "
                f"successes after {MAX_RUNS_PER_SCENE} runs"
            )


async def _run_t2i_mode(
    output_dir: Path,
    mode: str,
    num_per_scene: int,
    t2i_model_name: str,
    generations: list[dict],
) -> None:
    model = get_t2i_model(t2i_model_name)
    for scene_name in SCENES_TO_CONFIG_OVERRIDES:
        scene_spec = _load_scene_spec(scene_name)
        prompt = (
            _build_detailed_t2i_prompt(scene_spec)
            if mode == "t2i_detailed"
            else scene_spec.scene_as_natural_language
        )
        scene_dir = output_dir / scene_name / mode
        scene_dir.mkdir(parents=True, exist_ok=True)
        img_idx = 0
        for i in range(num_per_scene):
            for attempt in range(3):
                try:
                    url = await model.generate(prompt=prompt)
                    save_path = scene_dir / f"{img_idx:03d}.png"
                    await download_image(url, save_path)
                    break
                except Exception:
                    logger.exception(
                        f"[{mode}][{scene_name}] generation {i} "
                        f"attempt {attempt + 1}/3 failed"
                    )
                    await asyncio.sleep(2)
            else:
                logger.error(f"[{mode}][{scene_name}] generation {i} failed after 3 attempts")
                continue
            generations.append({
                "scene": scene_name,
                "mode": mode,
                "filepath": str(save_path.relative_to(output_dir)),
                "prompt": prompt,
                "poseblend_run_id": "",
                "edit_chain_idx": "",
            })
            img_idx += 1
            logger.info(f"[{mode}][{scene_name}] saved {save_path} ({img_idx}/{num_per_scene})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment data across scenes.")
    parser.add_argument(
        "--modes", nargs="+", choices=ALL_MODES, default=ALL_MODES,
        help="Generation modes to run (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/outputs"),
        help="Root output directory (default: experiments/outputs)",
    )
    parser.add_argument(
        "--min-successful", type=int, default=15,
        help="Target successful generations per scene (default: 15)",
    )
    parser.add_argument(
        "--edit-model", type=str, default=DEFAULT_EDIT_MODEL,
        help=f"Edit model for poseblend (default: {DEFAULT_EDIT_MODEL})",
    )
    parser.add_argument(
        "--t2i-model", type=str, default=DEFAULT_T2I_MODEL,
        help=f"T2I model for 0-shot generation (default: {DEFAULT_T2I_MODEL})",
    )
    args = parser.parse_args()

    async def _main() -> None:
        generations: list[dict] = []
        run_log: list[dict] = []
        for mode in ("t2i_simple", "t2i_detailed"):
            if mode in args.modes:
                await _run_t2i_mode(
                    args.output_dir, mode, args.min_successful,
                    args.t2i_model, generations,
                )
        if "poseblend" in args.modes:
            await _run_poseblend_mode(
                args.output_dir, args.min_successful, args.edit_model,
                generations, run_log,
            )
        _write_csv(args.output_dir / "generations.csv", GENERATION_FIELDS, generations)
        if run_log:
            _write_csv(args.output_dir / "poseblend_runs.csv", RUN_LOG_FIELDS, run_log)
        logger.info(f"Done — {len(generations)} generations, {len(run_log)} poseblend runs logged")

    asyncio.run(_main())
    