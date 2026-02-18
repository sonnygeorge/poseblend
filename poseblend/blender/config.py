import math

# --- Camera sampling ---
# Max random camera angles to try before accepting overlap
MAX_CAMERA_ANGLE_SAMPLES = 50
# Max times to retry a failed Cycles render call
MAX_RENDER_ATTEMPTS = 5
# Lower bound on camera elevation in radians (~7.5 deg, near-horizontal)
CAMERA_ELEVATION_MIN_RADS = math.radians(7.5)
# Upper bound on camera elevation in radians (~81 deg, near-overhead)
CAMERA_ELEVATION_MAX_RADS = math.radians(81)
# Mean of the normal distribution used to sample camera elevation (~31 deg)
CAMERA_ELEVATION_MEAN_RADS = math.radians(31)
# Std-dev of camera elevation sampling distribution (~12 deg spread)
CAMERA_ELEVATION_STD_RADS = math.radians(12)
# Blender render engine to use
RENDER_ENGINE = "CYCLES"

# --- Position contraction ---
# Density (object vol / envelope vol) at or above which no contraction is
# applied. The higher this is, the more aggressively scenes get contracted.
DENSITY_THRESHOLD = 0.35
# Floor on the contraction scalar so objects never collapse to a singularity
# (0–1; 0.3 means positions shrink to at most 30% of their original spread)
S_MIN = 0.3

# --- Airspace enforcement ---
# Fraction of the min camera distance used as the minimum surface-to-surface
# gap between objects (e.g. 0.1 = 10% of camera distance)
AIRSPACE_FACTOR = 0.1
# Number of pairwise-repulsion passes to resolve airspace violations
AIRSPACE_REPULSION_ITERATIONS = 5