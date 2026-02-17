# PoseBlend

PoseBlend: Compositional Action Scenes via Critic-Gated Localized Pose Editing of Blender Renders

> Text-to-image models perform well on common scenes but struggle with long-tail and out-of-distribution compositions. In particular, they fail to generate complex action scenes in which multiple entities must assume coherent roles, spatial relationships, and physically plausible poses—especially in rare configurations underrepresented in training data. We propose PoseBlend, an open-source image generation pipeline that (1) arranges and renders scenes of arbitrary Blender objects and (2) applies critic-gated localized diffusion edits to introduce realistic backgrounds and iteratively refine object poses. We demonstrate the system on atypical action-role configurations and show that PoseBlend consistently outperforms <SOTA MODEL>, achieving an <NUMBER>\% improvement in mean VQA score. Beyond improving generation quality, PoseBlend enables controlled synthesis of rare and parameterized action-role configurations, supporting synthetic data augmentation and systematic study of compositional generalization in vision-language models. The system is modular, model-agnostic, and released for reproducible research.

## How does it work?

![PoseBlend Pipeline](static/poseblend_flowchart.png)

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

To run, PoseBlend requires three things:

1. An input scene specification
2. A configuration of hyperparameters
3. A registry/repository of usable Blender objects

### Scene Specs

Input scene specifications are lists of visual requirements that double as a sequence of localized edits to be made once the objects have been spatially arranged and rendered in Blender.

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
