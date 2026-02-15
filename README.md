# PoseBlend

PoseBlend: Compositional Action Scenes via Critic-Gated Localized Pose Editing of Blender Renders

## Motivation

- One interesting flavor of scene compositionality is how actions compose with their role arguments.
- In this regard, T2I models are known to struggle with:
  1. Asymmetric bias in role assignments
  2. 3-or-more-argument action compositions (e.g., ditransitive verbs) with atypical arguments
- We propose a semi-procedural generation pipeline to controllably generate images of such scenes.

## Running PoseBlend

```bash
# Run the pipeline (headless, no GUI)
python main.py

# Run with real-time GUI
python main.py --gui

# Use a specific scene spec and/or config
python main.py --scene inputs/scenes/puma_shows_bird_to_person_car_nearby.yaml
python main.py --config inputs/configs/simple.yaml --scene inputs/scenes/some_scene.yaml

# Use a custom port for the GUI server (default: 8420)
python main.py --gui --port 9000
```

## Viewing Past Runs

```bash
# View a previous run's results in the GUI (by run directory)
python main.py --view outputs/20260214_224402
```

## Rebuilding the GUI

The backend serves the GUI from `gui/dist/`, so after making changes to the Svelte
frontend, you need to rebuild:

```bash
cd gui && npm run build && cd ..
```

To do a full reset (reinstall deps + rebuild):

```bash
cd gui && npm install && npm run build && cd ..
```

After rebuilding, hard-refresh the browser (`Cmd+Shift+R` / `Ctrl+Shift+R`) or close and
reopen the tab to ensure the new assets are loaded.

## Inputs

### Scene Specs

(specified by a human in YAML format)

```yaml
... TODO: Paste example here
```

### Configs

(specified by a human in YAML format)

```yaml
... TODO: Paste example here
```

### Blender Object Registry

TODO: ...

## Poseblend Pipeline Logic

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
