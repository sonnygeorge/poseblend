from loguru import logger

from poseblend.exceptions import NoSceneGoodEnoughError
from poseblend.run_context import RunContext
from poseblend.schema.run_data import BlenderScene, GateDecision, SceneRender


def select_renders(ctx: RunContext) -> list[SceneRender]:
    config = ctx.run_data.config

    # Compute scene_quality_score for each scene (mean of render scores)
    for scene in ctx.run_data.scenes:
        scores = [r.render_quality_score or 0.0 for r in scene.renders]
        scene.scene_quality_score = sum(scores) / len(scores) if scores else 0.0

    # Select the winning scene
    best_scene: BlenderScene = max(ctx.run_data.scenes, key=lambda s: s.scene_quality_score)
    best_scene.is_selected = True
    best_scene.gate_decision = GateDecision(
        is_passing=True,
        reason=(
            f"Got highest scene_quality_score "
            f"({best_scene.scene_quality_score:.3f})"
        ),
    )
    for scene in ctx.run_data.scenes:
        if scene is not best_scene:
            scene.gate_decision = GateDecision(
                is_passing=False,
                reason=(
                    f"Scene quality score of {scene.scene_quality_score:.3f} "
                    f"was lower than the winning score of "
                    f"{best_scene.scene_quality_score:.3f}"
                ),
            )

    # Set gate decisions for renders in the winning scene that don't already
    # have one (visibility-fail renders already have theirs from scoring)
    threshold = config.min_render_quality_score
    for render in best_scene.renders:
        if render.gate_decision is not None:
            continue
        score = render.render_quality_score or 0.0
        if score >= threshold:
            render.gate_decision = GateDecision(
                is_passing=True,
                reason=(
                    f"Render quality score of {score:.3f} is above "
                    f"the {threshold} threshold"
                ),
            )
        else:
            render.gate_decision = GateDecision(
                is_passing=False,
                reason=(
                    f"Render quality score of {score:.3f} is below "
                    f"the {threshold} threshold"
                ),
            )

    # Rank passing renders by score, take top k
    ranked_passing = sorted(
        [r for r in best_scene.renders if r.gate_decision and r.gate_decision.is_passing],
        key=lambda r: r.render_quality_score or 0.0,
        reverse=True,
    )

    if not ranked_passing:
        raise NoSceneGoodEnoughError(
            f"No renders in scene {best_scene.scene_id} meet the minimum quality "
            f"threshold ({threshold}). Best render score: "
            f"{max((r.render_quality_score or 0.0 for r in best_scene.renders), default=0.0):.3f}"
        )

    selected = ranked_passing[: config.num_edit_chains]
    logger.info(
        f"Selected {len(selected)}/{len(best_scene.renders)} renders for editing from "
        f"scene {best_scene.scene_id} (scores: "
        f"{[f'{r.render_quality_score:.3f}' for r in selected]})"
    )
    return selected
