<script>
  import { runData } from "../stores.js";
  import { imageUrl, gateShadow } from "../utils.js";
  import ImageCard from "../lib/ImageCard.svelte";
  import ImageGroup from "../lib/ImageGroup.svelte";

  let data = $derived($runData);
  let editChains = $derived(data?.edit_chains || []);
  let runId = $derived(data?.run_id || "");

  function buildAttemptItems(attempts, chainIdx, editIdx) {
    return (attempts || []).map((a, i) => ({
      key: `chain-${chainIdx}-edit-${editIdx}-attempt-${i}`,
      imageSrc: imageUrl(a.after_img_path, runId),
      gateDecision: a.gate_decision,
      metadata: {
        modelUsed: a.model_used,
        seed: a.seed,
        promptUsed: a.prompt_used,
        criticInvocations: a.critic_invocations,
        gateDecision: a.gate_decision,
      },
    }));
  }

  function editGroupGateDecision(attempts) {
    if (!attempts || attempts.length === 0) return null;
    const last = attempts[attempts.length - 1];
    if (last.gate_decision?.is_passing) return last.gate_decision;
    // Still in progress if we haven't hit max attempts and last failed
    return last.gate_decision;
  }
</script>

<div class="edit-chains">
  <h2>Edits</h2>
  {#if editChains.length === 0}
    <p class="waiting">Waiting for edit chains...</p>
  {:else}
    <div class="chains-row">
      {#each editChains as chain, chainIdx (chainIdx)}
        <div
          class="chain-column"
          style="box-shadow: {gateShadow(chain.gate_decision)}"
        >
          <span class="chain-label">Chain {chainIdx}</span>

          <ImageCard
            cardKey={`chain-${chainIdx}-starting-render`}
            imageSrc={imageUrl(chain.starting_render_path, runId)}
            gateDecision={{ is_passing: true, reason: "Selected render" }}
            metadata={{ gateDecision: { is_passing: true, reason: "Selected render" } }}
            size="small"
          />

          {#each chain.edits as attempts, editIdx (editIdx)}
            <div class="edit-row">
              <span class="edit-label">{editIdx === 0 ? "Background" : `Edit ${editIdx}`}</span>
              <ImageGroup
                items={buildAttemptItems(attempts, chainIdx, editIdx)}
                gateDecision={editGroupGateDecision(attempts)}
                imageSize="small"
              />
            </div>
          {/each}

          {#if chain.candidate_final_img_path}
            <div class="final-row">
              <span class="edit-label">Final</span>
              <ImageCard
                cardKey={`chain-${chainIdx}-final`}
                imageSrc={imageUrl(chain.candidate_final_img_path, runId)}
                gateDecision={chain.gate_decision}
                metadata={{
                  gateDecision: chain.gate_decision,
                  criticInvocations: chain.final_critic_invocations,
                }}
              />
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .edit-chains {
    padding: 12px 20px 12px 20px;
  }
  h2 {
    margin: 0 0 16px 0;
    font-size: 1rem;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .chains-row {
    display: flex;
    gap: 20px;
    overflow-x: auto;
    padding-bottom: 8px;
  }
  .chain-column {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px;
    border-radius: 10px;
    background:rgb(38, 38, 38);
    min-width: 180px;
    flex-shrink: 0;
    margin: 4px;
  }
  .chain-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .edit-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .edit-label {
    font-size: 0.7rem;
    color: #777;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .final-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: flex-start;
  }
  .waiting {
    color: #666;
    font-style: italic;
  }
</style>
