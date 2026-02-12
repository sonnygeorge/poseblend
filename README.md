# PoseBlend

PoseBlend: Generating Compositional Action Scenes via Critic-Gated Localized Editing of Blender Renders

## Motivation

- One interesting flavor of scene compositionality is how actions compose with their role arguments.
- In this regard, T2I models are known to struggle with:
  1. Asymmetric bias in role assignments
  2. 3-or-more-argument action compositions (e.g., ditransitive verbs) with atypical arguments
- We propose a semi-procedural generation pipeline to controllably generate images of such scenes.

## Input Scene Spec

(specified by a human in YAML format)

```yaml
scene_as_natural_language: A puma shows a bird to a person. A dog is nearby.
action: shows
role_assignments:
  shower: puma
  shown: bird
  shown_to: person
  nearby: dog
localized_edits:
  - region_contains:
      - shower
      - shown
    requirements:
      - "{shower}'s body is oriented towards {shown_to}."
      - "{shower}'s gaze is looking at {shown}."
      - "{shower}'s arm is outstretched."
      - "{shown} is perched on {shower}'s arm."
  - region_contains:
      - shown_to
    requirements:
      - "{shown_to} is close to {shower} with their body oriented towards them."
      - "{shown_to}'s gaze is looking at {shown}."
  - region_contains:
      - nearby
    requirements:
      - "{nearby} is in the scene periphery, paying no attention to {shower}, {shown}, or {shown_to}."
```

## Pipeline Steps

### Step 1: Generate Blender Scene Renders

1. Generate $n_{bs}$ blender scene params
2. For each:
    - Setup a blender scene
    - Render $n_p$ povs, calculating segment boundaries for each edit region
3. Get `render_quality_scores` for all renders
4. Select top $k$ renders from blender scene w/ highest mean `render_quality_score`
5. Error out if mean `render_quality_score` of these top $k$ renders is less than $t_r$

### Step 2: Apply Localized Edits To Get Final Images

1. For each render:
    - for i in `max_edit_attempts`:
        - Do background edit
        - Perform background edit requirement checks and as soon as one fails, either continue to next edit attempt or, if on last attempt, mark render as failure and abandon editing it further.
        - If all checks pass, break.
    - For each localized edit:
        - Do this edit's requirements pass? If yes, continue (no need to apply edit)
        - For i in `max_edit_attempts`:
            - Apply edit (using drawn-on segmentation boundary?)
            - Loop through accumulating requirement checks and as soon as one fails, either continue to next edit attempt or, if on last attempt, mark render as failure and abandon editing it.
            - If all checks pass, break and continue to next edit.
    - If all edits successful, save last edited image as final image.

NOTE: Even higher avg. quality can be achieved by ranking/dropping some % from final set of images based on some aggregation of confidence score across last round of checks.

## TODO

- Scene param gen step
  - Generate N blender scene params asynchronously
- Blender step
  - Figure out if multiple blender scene rendering processes can be run in parallel
  - Move blender rendering code into blender/, decide how to share data b/w processes, and write the subprocess running util
- Scene & render choice step
  - Critic process for scoring renders...
  
## GUI Notes

States:
- "Generating blender scene params"
- "Deciding which scene to use"
- Display edit chains and monitor updates...