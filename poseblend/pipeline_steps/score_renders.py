import asyncio

from poseblend.run_context import RunContext
from poseblend.schema.run_data import CriticInvocation, GateDecision, SceneRender
from poseblend.utils import build_visibility_requirements, invoke_critic

VISIBILITY_GATE_THRESHOLD = 0.4


def _build_edit_requirements(ctx: RunContext) -> list[str]:
    hydrated = ctx.run_data.scene_spec.get_hydrated_edit_requirements()
    return [req for group in hydrated for req in group]


async def _score_single_render(
    ctx: RunContext,
    render: SceneRender,
    visibility_requirements: list[str],
    edit_requirements: list[str],
) -> None:
    if render.image_path is None:
        msg = f"Render {render.render_id} has no image_path"
        raise ValueError(msg)

    # Phase 1: visibility checks (hard gate, short-circuit on first failure)
    visibility_results = []
    for req in visibility_requirements:
        result = await invoke_critic(ctx, render.image_path, req)
        render.critic_invocations.append(
            CriticInvocation(requirement=req, result=result)
        )
        visibility_results.append(result)
        if result.normalized_score < VISIBILITY_GATE_THRESHOLD:
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
            return
    visibility_scores = [r.normalized_score for r in visibility_results]

    # Phase 2: edit requirement checks
    edit_results = await asyncio.gather(*[
        invoke_critic(ctx, render.image_path, req)
        for req in edit_requirements
    ])
    for req, result in zip(edit_requirements, edit_results):
        render.critic_invocations.append(
            CriticInvocation(requirement=req, result=result)
        )
    edit_scores = [r.normalized_score for r in edit_results]

    all_scores = visibility_scores + edit_scores
    render.render_quality_score = sum(all_scores) / len(all_scores)


async def score_all_renders(ctx: RunContext) -> None:
    visibility_reqs = build_visibility_requirements(ctx)
    edit_reqs = _build_edit_requirements(ctx)

    all_renders = [
        render
        for scene in ctx.run_data.scenes
        for render in scene.renders
    ]

    await asyncio.gather(*[
        _score_single_render(ctx, render, visibility_reqs, edit_reqs)
        for render in all_renders
    ])
