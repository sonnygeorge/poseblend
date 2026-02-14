export function gateColor(gateDecision) {
  if (gateDecision == null) return "#4a90d9"; // blue — pending
  return gateDecision.is_passing ? "#22a45c" : "#f05545"; // green / red
}

export function gateShadow(gateDecision) {
  const color = gateColor(gateDecision);
  return `0 0 1px 2px ${color}B3, 0 0 4px 3px ${color}35, 0 0 5px 5px ${color}15`;
}

export function gateLabel(gateDecision) {
  if (gateDecision == null) return "Pending";
  return gateDecision.is_passing ? "Passed" : "Failed";
}

export function imageUrl(filePath, runId) {
  if (!filePath) return "";
  const str = String(filePath);
  // Strip everything up to and including the run_id directory
  const marker = `${runId}/`;
  const idx = str.indexOf(marker);
  if (idx !== -1) {
    return `/files/${str.slice(idx + marker.length)}`;
  }
  // Fallback: try outputs/*/...
  const outputsIdx = str.indexOf("outputs/");
  if (outputsIdx !== -1) {
    const afterOutputs = str.slice(outputsIdx + "outputs/".length);
    const slashIdx = afterOutputs.indexOf("/");
    if (slashIdx !== -1) {
      return `/files/${afterOutputs.slice(slashIdx + 1)}`;
    }
  }
  return `/files/${str}`;
}

export function formatScore(score) {
  if (score == null) return "—";
  return score.toFixed(3);
}

export function likertLabel(score) {
  const labels = {
    1: "Not at all",
    2: "Slightly",
    3: "Moderately",
    4: "Mostly",
    5: "Clearly",
  };
  return labels[score] || String(score);
}
