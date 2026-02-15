import asyncio

from poseblend.run_context import RunContext
from poseblend.schema.run_data import CriticInvocation, GateDecision, SceneRender
from poseblend.utils import build_single_object_requirements, derive_seed, invoke_critic


def _build_edit_requirements(ctx: RunContext) -> list[str]:
    hydrated = ctx.run_data.scene_spec.get_hydrated_edit_requirements()
    return [req for group in hydrated for req in group]


async def _score_single_render(
    ctx: RunContext,
    render: SceneRender,
    single_object_requirements: list[str],
    edit_requirements: list[str],
    scene_seed: int | None = None,
) -> None:
    if render.image_path is None:
        msg = f"Render {render.render_id} has no image_path"
        raise ValueError(msg)

    render_seed = derive_seed(scene_seed, render.render_id) if scene_seed is not None else None
    n_single = len(single_object_requirements)

    # Phase 1: single-object checks (hard gate, short-circuit on first failure)
    single_object_results = []
    for req_idx, req in enumerate(single_object_requirements):
        critic_seed = derive_seed(render_seed, req_idx + 1) if render_seed is not None else None
        result = await invoke_critic(ctx, render.image_path, req, seed=critic_seed)
        render.critic_invocations.append(
            CriticInvocation(requirement=req, result=result)
        )
        single_object_results.append(result)
        if result.normalized_score < ctx.run_data.config.single_object_check_threshold:
            score_label = result.score.name.lower().replace("_", " ")
            threshold = ctx.run_data.config.min_render_quality_score
            render.render_quality_score = 0.0
            render.gate_decision = GateDecision(
                is_passing=False,
                reason=(
                    f'The requirement, "{req}" was deemed {score_label}, '
                    f"resulting in a quality score of 0.0, which is below the "
                    f'{threshold} threshold. Reasoning: "{result.reasoning}"'
                ),
            )
            ctx.on_run_data_changed()
            return
    single_object_scores = [r.normalized_score for r in single_object_results]

    # Phase 2: edit requirement checks
    edit_results = await asyncio.gather(*[
        invoke_critic(
            ctx,
            render.image_path,
            req,
            seed=derive_seed(render_seed, n_single + req_idx + 1) if render_seed is not None else None,
        )
        for req_idx, req in enumerate(edit_requirements)
    ])
    for req, result in zip(edit_requirements, edit_results):
        render.critic_invocations.append(
            CriticInvocation(requirement=req, result=result)
        )
    edit_scores = [r.normalized_score for r in edit_results]

    all_scores = single_object_scores + edit_scores
    render.render_quality_score = sum(all_scores) / len(all_scores)
    ctx.on_run_data_changed()


async def score_all_renders(ctx: RunContext) -> None:
    single_object_reqs = build_single_object_requirements(ctx)
    edit_reqs = _build_edit_requirements(ctx)

    render_scene_seed_pairs = [
        (render, scene.seed)
        for scene in ctx.run_data.scenes
        for render in scene.renders
    ]

    await asyncio.gather(*[
        _score_single_render(ctx, render, single_object_reqs, edit_reqs, scene_seed=seed)
        for render, seed in render_scene_seed_pairs
    ])
