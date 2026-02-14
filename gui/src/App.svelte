<script>
  import { runData } from "./stores.js";
  import RenderSelection from "./panes/RenderSelection.svelte";
  import EditChains from "./panes/EditChains.svelte";
  import DetailSidebar from "./lib/DetailSidebar.svelte";

  let data = $derived($runData);

  // --- Horizontal divider (top/bottom pane split) ---
  let topPaneFraction = $state(0.5);
  let draggingH = $state(false);
  let mainContentEl;

  function onHDragStart(e) {
    e.preventDefault();
    draggingH = true;
    window.addEventListener("mousemove", onHDragMove);
    window.addEventListener("mouseup", onHDragEnd);
  }

  function onHDragMove(e) {
    if (!mainContentEl) return;
    const rect = mainContentEl.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const fraction = Math.min(Math.max(y / rect.height, 0.15), 0.85);
    topPaneFraction = fraction;
  }

  function onHDragEnd() {
    draggingH = false;
    window.removeEventListener("mousemove", onHDragMove);
    window.removeEventListener("mouseup", onHDragEnd);
  }

  // --- Vertical divider (main content / sidebar split) ---
  let sidebarWidth = $state(340);
  let draggingV = $state(false);

  function onVDragStart(e) {
    e.preventDefault();
    draggingV = true;
    window.addEventListener("mousemove", onVDragMove);
    window.addEventListener("mouseup", onVDragEnd);
  }

  function onVDragMove(e) {
    const newWidth = window.innerWidth - e.clientX;
    sidebarWidth = Math.min(Math.max(newWidth, 200), 700);
  }

  function onVDragEnd() {
    draggingV = false;
    window.removeEventListener("mousemove", onVDragMove);
    window.removeEventListener("mouseup", onVDragEnd);
  }

  let isDragging = $derived(draggingH || draggingV);
</script>

<div class="app" class:dragging={isDragging}>
  <div class="main-content" bind:this={mainContentEl}>
    {#if !data}
      <div class="connecting">
        <p>Connecting to PoseBlend pipeline...</p>
      </div>
    {:else}
      <div class="pane top-pane" style="flex: {topPaneFraction}">
        <RenderSelection />
      </div>
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="h-divider" onmousedown={onHDragStart}></div>
      <div class="pane bottom-pane" style="flex: {1 - topPaneFraction}">
        <EditChains />
      </div>
    {/if}
  </div>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="v-divider" onmousedown={onVDragStart}></div>
  <div class="sidebar-wrapper" style="width: {sidebarWidth}px; min-width: {sidebarWidth}px;">
    <DetailSidebar />
  </div>
</div>

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
      sans-serif;
    background:rgb(24, 24, 24);
    color: #cdd6f4;
    overflow: hidden;
  }
  .app {
    display: flex;
    height: 100vh;
    width: 100vw;
  }
  .app.dragging {
    cursor: grabbing;
    user-select: none;
  }
  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
    position: relative;
    z-index: 1;
  }
  .sidebar-wrapper {
    position: relative;
    z-index: 10;
  }
  .pane {
    overflow-y: auto;
    overflow-x: hidden;
    min-height: 0;
    position: relative;
  }
  .bottom-pane {
    z-index: 2;
    box-shadow: 0 -10px 20px rgba(0, 0, 0, 0.32), 0 -25px 60px rgba(0, 0, 0, 0.27), 0 -50px 140px rgba(0, 0, 0, 0.2);
  }
  .h-divider {
    height: 6px;
    flex-shrink: 0;
    cursor: row-resize;
    background: linear-gradient(to right, transparent, #444, transparent);
    background-size: 100% 2px;
    background-repeat: no-repeat;
    background-position: center;
    margin: 0 16px;
    transition: background-color 0.15s;
  }
  .h-divider:hover {
    background-color: rgba(255, 255, 255, 0.05);
  }
  .v-divider {
    width: 6px;
    flex-shrink: 0;
    cursor: col-resize;
    background: linear-gradient(to bottom, transparent, #333, transparent);
    background-size: 2px 100%;
    background-repeat: no-repeat;
    background-position: center;
    transition: background-color 0.15s;
    z-index: 11;
  }
  .v-divider:hover {
    background-color: rgba(255, 255, 255, 0.05);
  }
  .connecting {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #666;
    font-style: italic;
    font-size: 1.1rem;
  }
</style>
