import asyncio
import random
import shutil
from pathlib import Path

from loguru import logger

from poseblend.run_context import RunContext
from poseblend.schema.inputs.config import AttemptRangeModelSelectionProbs
from poseblend.schema.run_data import (
    AttemptedEdit,
    CriticInvocation,
    EditChain,
    GateDecision,
    SceneRender,
)
from poseblend.utils import (
    build_single_object_requirements,
    derive_seed,
    derive_seeds,
    download_image,
    list_grammatically,
    image_path_to_data_uri,
    invoke_critic,
    sample_from_discrete_distribution,
)

EDIT_REQUIREMENT_PASS_THRESHOLD = 0.75  # Likert >= 4 (mostly/clearly satisfied)
EDIT_REQUIREMENT_RECHECK_THRESHOLD = 0.5  # Relaxed bar for previously-passed reqs
BG_ONLY_REQS = [
    "There is nothing utterly and egregiously wrong about the perspective from the pov of "
    "the camera. (DO NOT worry about the plausibility of spatial relationships b/w "
    "objects! Things can be in weird positions w.r.t. each other. However, if object "
    "scale and perspective geometries are seriously messed up, that's a problem.)"
]


def _get_model_probs_for_attempt(
    schedule: list[AttemptRangeModelSelectionProbs],
    attempt_num: int,
) -> dict[str, float]:
    for entry in schedule:
        if entry.from_attempt <= attempt_num and (
            entry.to_attempt is None or attempt_num <= entry.to_attempt
        ):
            return entry.model_probs
    msg = f"No model selection probs found for attempt {attempt_num}"
    raise ValueError(msg)


def _build_background_prompt(background_str: str, objects_str: str) -> str:
    return (
        "Making sure it coheres with the proportions and positions of existing objects, "
        f"fill in the empty space with a {background_str} scene backdrop. Use minimal, "
        "not-too-busy textures. Prefer a neutral color scheme that differs from the colors"
        " of the existing objects so that they stand out from the background."
    )


def _build_localized_edit_prompt(objects_to_edit_str: str, requirements: list[str]) -> str:
    requirements_str = list_grammatically(requirements, enumerate=True)
    return (
        f"Keeping everything else THE EXACT SAME, update the physical state(s)/pose(s) of ONLY" 
        f" {objects_to_edit_str} without changing its/their size or location " 
        f"such that: {requirements_str}"
    )

async def _check_requirements(
    ctx: RunContext,
    image_path: Path,
    requirement_groups: list[tuple[list[str], float]],
    base_seed: int | None = None,
) -> tuple[GateDecision, list[CriticInvocation]]:
    invocations: list[CriticInvocation] = []
    global_req_idx = 0
    for requirements, threshold in requirement_groups:
        for req in requirements:
            global_req_idx += 1
            critic_seed = derive_seed(base_seed, global_req_idx) if base_seed is not None else None
            result = await invoke_critic(ctx, image_path, req, seed=critic_seed)
            invocations.append(CriticInvocation(requirement=req, result=result))
            if result.normalized_score < threshold:
                decision = GateDecision(
                    is_passing=False,
                    reason=(
                        f'Judged to not meet the requirement: "{req}". '
                        f'Reasoning: "{result.reasoning}"'
                    ),
                )
                return decision, invocations
    return GateDecision(is_passing=True, reason="All requirements passed"), invocations


async def _run_single_edit_chain(
    ctx: RunContext,
    edit_chain: EditChain,
    chain_idx: int,
    chain_seed: int | None,
) -> None:
    config = ctx.run_data.config
    scene_spec = ctx.run_data.scene_spec
    max_attempts = config.max_edit_attempts
    single_object_threshold = config.single_object_check_threshold
    chain_dir = ctx.run_data.run_dir / "edit_chains" / str(chain_idx)

    single_object_reqs = build_single_object_requirements(ctx)
    hydrated_edit_reqs = scene_spec.get_hydrated_edit_requirements()
    unique_objects = sorted(set(scene_spec.role_assignments.values()))
    objects_str = list_grammatically([f"the {obj}" for obj in unique_objects])

    current_img_path = edit_chain.starting_render_path
    previously_passed_edit_reqs: list[str] = []
    tick = 0

    def _next_seed() -> int | None:
        nonlocal tick
        tick += 1
        if chain_seed is None:
            return None
        return derive_seed(chain_seed, tick)

    def _sub_seed(attempt_seed: int | None, n: int) -> int | None:
        if attempt_seed is None:
            return None
        return derive_seed(attempt_seed, n)

    # --- Edit 0: Background edit ---
    bg_attempts: list[AttemptedEdit] = []
    edit_chain.edits.append(bg_attempts)

    for attempt_num in range(1, max_attempts + 1):
        attempt_seed = _next_seed()
        bg_str = random.Random(_sub_seed(attempt_seed, 1)).choice(config.background_strs)

        model_probs = _get_model_probs_for_attempt(
            config.edit_model_selection_schedule, attempt_num
        )
        model_name = sample_from_discrete_distribution(
            model_probs, seed=_sub_seed(attempt_seed, 2)
        )

        prompt = _build_background_prompt(bg_str, objects_str)
        source_uri = image_path_to_data_uri(current_img_path)
        edit_model = ctx.get_image_edit_model(model_name)
        generation_seed = _sub_seed(attempt_seed, 3)
        result_url = await edit_model.edit(
            prompt=prompt, image_urls=[source_uri], seed=generation_seed
        )

        after_path = chain_dir / f"edit_0_attempt_{attempt_num}.png"
        await download_image(result_url, after_path)

        decision, invocations = await _check_requirements(
            ctx,
            after_path,
            [
                (BG_ONLY_REQS, EDIT_REQUIREMENT_PASS_THRESHOLD),
                (single_object_reqs, single_object_threshold),
            ],
            base_seed=_sub_seed(attempt_seed, 4),
        )
        bg_attempts.append(AttemptedEdit(
            seed=attempt_seed,
            before_img_path=current_img_path,
            after_img_path=after_path,
            prompt_used=prompt,
            model_used=model_name,
            critic_invocations=invocations,
            
            gate_decision=decision,
        ))
        ctx.on_run_data_changed()

        if decision.is_passing:
            current_img_path = after_path
            logger.debug(
                f"Edit chain {chain_idx}: background edit passed on attempt {attempt_num}"
            )
            break

        logger.debug(
            f"Edit chain {chain_idx}: background edit failed attempt {attempt_num}"
            f"/{max_attempts} — {decision.reason}"
        )
    else:
        edit_chain.gate_decision = GateDecision(
            is_passing=False,
            reason=(
                f"Background edit failed after {max_attempts} attempts. "
                f"Last failure: {bg_attempts[-1].gate_decision.reason}"
            ),
        )
        ctx.on_run_data_changed()
        return

    # Strip out BG_ONLY_REQS invocations before carrying forward. BG_ONLY_REQS are
    # requirements specific to the background edit and are not requirements for localized
    # edits. If a localized edit gets skipped (because its requirements are already met),
    # these carried-forward invocations get attached to the skipped edit's record. Including
    # BG_ONLY_REQS there would incorrectly present them as requirements that the localized
    # edit "passed," when they were never requirements for such edits in the first place.
    bg_only_req_set = set(BG_ONLY_REQS)
    persisting_invocations_if_next_edit_skipped = [
        inv for inv in bg_attempts[-1].critic_invocations
        if inv.requirement not in bg_only_req_set
    ]

    # --- Edits 1..N: Localized edits ---
    for edit_idx, (edit_spec, edit_reqs) in enumerate(
        zip(scene_spec.localized_edits, hydrated_edit_reqs), start=1
    ):
        edit_attempts: list[AttemptedEdit] = []
        edit_chain.edits.append(edit_attempts)

        # Pre-check: are the new edit requirements already satisfied?
        pre_check_seed = _next_seed()
        pre_decision, pre_invocations = await _check_requirements(
            ctx, current_img_path, [(edit_reqs, EDIT_REQUIREMENT_PASS_THRESHOLD)],
            base_seed=pre_check_seed,
        )
        if pre_decision.is_passing:
            previously_passed_edit_reqs.extend(edit_reqs)
            combined_invocations = persisting_invocations_if_next_edit_skipped + pre_invocations
            edit_attempts.append(AttemptedEdit(
                seed=None,
                before_img_path=current_img_path,
                after_img_path=current_img_path,
                prompt_used=None,
                model_used=None,
                critic_invocations=combined_invocations,
                gate_decision=GateDecision(
                    is_passing=True,
                    reason="Edit skipped — new requirements already satisfied",
                ),
            ))
            persisting_invocations_if_next_edit_skipped = combined_invocations
            ctx.on_run_data_changed()
            logger.debug(
                f"Edit chain {chain_idx}: localized edit {edit_idx} skipped — "
                "requirements already satisfied"
            )
            continue

        region_objects = [scene_spec.role_assignments[role] for role in edit_spec.region_contains]
        object_names = list_grammatically([f"the {obj}" for obj in region_objects])

        for attempt_num in range(1, max_attempts + 1):
            attempt_seed = _next_seed()

            model_probs = _get_model_probs_for_attempt(
                config.edit_model_selection_schedule, attempt_num
            )
            model_name = sample_from_discrete_distribution(
                model_probs, seed=_sub_seed(attempt_seed, 1)
            )

            prompt = _build_localized_edit_prompt(object_names, edit_reqs)
            source_uri = image_path_to_data_uri(current_img_path)
            edit_model = ctx.get_image_edit_model(model_name)
            generation_seed = _sub_seed(attempt_seed, 2)
            result_url = await edit_model.edit(
                prompt=prompt, image_urls=[source_uri], seed=generation_seed
            )

            after_path = chain_dir / f"edit_{edit_idx}_attempt_{attempt_num}.png"
            await download_image(result_url, after_path)

            decision, invocations = await _check_requirements(
                ctx,
                after_path,
                [
                    (single_object_reqs, single_object_threshold),
                    (previously_passed_edit_reqs, EDIT_REQUIREMENT_RECHECK_THRESHOLD),
                    (edit_reqs, EDIT_REQUIREMENT_PASS_THRESHOLD),
                ],
                base_seed=_sub_seed(attempt_seed, 3),
            )
            edit_attempts.append(AttemptedEdit(
                seed=attempt_seed,
                before_img_path=current_img_path,
                after_img_path=after_path,
                prompt_used=prompt,
                model_used=model_name,
                critic_invocations=invocations,
                gate_decision=decision,
            ))
            ctx.on_run_data_changed()

            if decision.is_passing:
                current_img_path = after_path
                previously_passed_edit_reqs.extend(edit_reqs)
                persisting_invocations_if_next_edit_skipped = invocations
                logger.debug(
                    f"Edit chain {chain_idx}: localized edit {edit_idx} passed on "
                    f"attempt {attempt_num}"
                )
                break

            logger.debug(
                f"Edit chain {chain_idx}: localized edit {edit_idx} failed attempt "
                f"{attempt_num}/{max_attempts} — {decision.reason}"
            )
        else:
            edit_chain.gate_decision = GateDecision(
                is_passing=False,
                reason=(
                    f"Localized edit {edit_idx} failed after {max_attempts} attempts. "
                    f"Last failure: {edit_attempts[-1].gate_decision.reason}"
                ),
            )
            ctx.on_run_data_changed()
            return

    edit_chain.candidate_final_img_path = current_img_path

    # Check final requirements (if any)
    hydrated_final_reqs = scene_spec.get_hydrated_final_requirements()
    if hydrated_final_reqs:
        final_check_seed = _next_seed()
        decision, invocations = await _check_requirements(
            ctx, current_img_path, [(hydrated_final_reqs, EDIT_REQUIREMENT_PASS_THRESHOLD)],
            base_seed=final_check_seed,
        )
        edit_chain.final_critic_invocations = invocations
        if decision.is_passing:
            edit_chain.gate_decision = decision
        else:
            edit_chain.gate_decision = GateDecision(
                is_passing=False,
                reason=f"Final requirements check failed. {decision.reason}",
            )
    else:
        edit_chain.gate_decision = GateDecision(
            is_passing=True,
            reason="All edits completed successfully",
        )
    ctx.on_run_data_changed()


async def perform_all_edits(
    ctx: RunContext,
    selected_renders: list[SceneRender],
) -> None:
    chain_seeds = derive_seeds(ctx.run_data.config.seed, len(selected_renders))
    # Eagerly create edit chains on run_data so GUI can see them immediately
    ctx.run_data.edit_chains = [
        EditChain(starting_render_path=render.image_path, edits=[], candidate_final_img_path=None)
        for render in selected_renders
    ]
    ctx.on_run_data_changed()

    await asyncio.gather(*[
        _run_single_edit_chain(ctx, edit_chain, chain_idx, seed)
        for chain_idx, (seed, edit_chain) in enumerate(
            zip(chain_seeds, ctx.run_data.edit_chains)
        )
    ])

    # Copy successful final images to a dedicated directory
    final_imgs_dir = ctx.run_data.run_dir / "final_imgs"
    final_imgs_dir.mkdir(parents=True, exist_ok=True)
    for chain_idx, ec in enumerate(ctx.run_data.edit_chains):
        if ec.gate_decision and ec.gate_decision.is_passing and ec.candidate_final_img_path:
            dest = final_imgs_dir / f"chain_{chain_idx}.png"
            shutil.copy2(ec.candidate_final_img_path, dest)

    n_success = sum(
        1 for ec in ctx.run_data.edit_chains if ec.gate_decision and ec.gate_decision.is_passing
    )
    n_fail = sum(
        1 for ec in ctx.run_data.edit_chains
        if ec.gate_decision and not ec.gate_decision.is_passing
    )
    logger.info(f"Edit chains complete: {n_success} succeeded, {n_fail} failed")
