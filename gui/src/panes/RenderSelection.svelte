<script>
  import { runData } from "../stores.js";
  import { imageUrl, formatScore } from "../utils.js";
  import ImageGroup from "../lib/ImageGroup.svelte";

  let data = $derived($runData);
  let scenes = $derived(data?.scenes || []);
  let runId = $derived(data?.run_id || "");

  function effectiveRenderGate(scene, render) {
    // If the scene was not selected, all its renders are effectively rejected
    if (scene.gate_decision && !scene.gate_decision.is_passing) {
      return { is_passing: false, reason: "Scene was not selected" };
    }
    return render.gate_decision;
  }

  function buildRenderItems(scene) {
    return (scene.renders || []).map((r, i) => ({
      key: `render-${scene.scene_id}-${r.render_id}`,
      imageSrc: imageUrl(r.image_path, runId),
      gateDecision: effectiveRenderGate(scene, r),
      metadata: {
        renderQualityScore: r.render_quality_score,
        criticInvocations: r.critic_invocations,
        gateDecision: effectiveRenderGate(scene, r),
      },
    }));
  }
</script>

<div class="render-selection">
  <h2>Scenes</h2>
  {#if scenes.length === 0}
    <p class="waiting">Waiting for scenes...</p>
  {:else}
    <div class="scenes-list">
      {#each scenes as scene (scene.scene_id)}
        <ImageGroup
          items={buildRenderItems(scene)}
          gateDecision={scene.gate_decision}
          label="Scene {scene.scene_id}{scene.seed != null ? ` (seed: ${scene.seed})` : ''}"
          groupMetadata={{
            sceneId: scene.scene_id,
            seed: scene.seed,
            sceneQualityScore: scene.scene_quality_score,
            gateDecision: scene.gate_decision,
            isSelected: scene.is_selected,
            modelUsed: $runData?.config?.blender_lm,
            promptUsed: scene.prompt_used,
          }}
        />
      {/each}
    </div>
  {/if}
</div>

<style>
  .render-selection {
    padding: 12px 20px 12px 20px;
  }
  h2 {
    margin: 0 0 16px 0;
    font-size: 1rem;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .scenes-list {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .waiting {
    color: #666;
    font-style: italic;
  }
</style>
