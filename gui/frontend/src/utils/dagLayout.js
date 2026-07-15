/**
 * DAG layout algorithm: assigns x,y positions to subtask nodes.
 *
 * Uses topological sort + layer assignment (Coffman-Graham style).
 * Outputs a flat list of { node, x, y, width, height } objects
 * plus a list of edge { from, to, fromX, fromY, toX, toY } objects.
 */

const NODE_W = 180;
const NODE_H = 80;
const H_GAP = 48;   // horizontal gap between parallel nodes
const V_GAP = 72;   // vertical gap between layers

export function layoutDag(subtasks) {
  if (!subtasks || subtasks.length === 0) return { nodes: [], edges: [] };

  // Build adjacency
  const byId = {};
  subtasks.forEach((s) => { byId[s.id] = s; });

  // Topological sort → assign layers
  const layers = {};  // id → layer index
  const maxLayer = { val: 0 };

  function assignLayer(id, visited = new Set()) {
    if (layers[id] !== undefined) return layers[id];
    if (visited.has(id)) return 0;
    visited.add(id);
    const task = byId[id];
    if (!task || !task.depends_on?.length) {
      layers[id] = 0;
      return 0;
    }
    const maxDep = Math.max(...task.depends_on.map((d) => assignLayer(d, visited)));
    layers[id] = maxDep + 1;
    if (layers[id] > maxLayer.val) maxLayer.val = layers[id];
    return layers[id];
  }

  subtasks.forEach((s) => assignLayer(s.id));

  // Group by layer
  const byLayer = {};
  subtasks.forEach((s) => {
    const l = layers[s.id] || 0;
    if (!byLayer[l]) byLayer[l] = [];
    byLayer[l].push(s);
  });

  // Calculate canvas size
  const numLayers = maxLayer.val + 1;
  const maxWidth = Math.max(...Object.values(byLayer).map((arr) => arr.length));

  // Assign positions
  const nodes = [];
  const nodePos = {};  // id → { cx, cy } (center)

  for (let layer = 0; layer < numLayers; layer++) {
    const row = byLayer[layer] || [];
    const totalRowW = row.length * NODE_W + (row.length - 1) * H_GAP;
    const startX = (maxWidth * (NODE_W + H_GAP) - totalRowW) / 2;

    row.forEach((s, i) => {
      const x = startX + i * (NODE_W + H_GAP);
      const y = layer * (NODE_H + V_GAP);
      nodePos[s.id] = { cx: x + NODE_W / 2, cy: y + NODE_H / 2 };
      nodes.push({ id: s.id, x, y, width: NODE_W, height: NODE_H, layer });
    });
  }

  // Build edges
  const edges = [];
  subtasks.forEach((s) => {
    (s.depends_on || []).forEach((depId) => {
      if (nodePos[depId] && nodePos[s.id]) {
        edges.push({
          from: depId,
          to: s.id,
          fromX: nodePos[depId].cx,
          fromY: nodePos[depId].cy + NODE_H / 2,
          toX: nodePos[s.id].cx,
          toY: nodePos[s.id].cy - NODE_H / 2,
        });
      }
    });
  });

  // Canvas dimensions
  const canvasW = maxWidth * (NODE_W + H_GAP);
  const canvasH = numLayers * (NODE_H + V_GAP);

  return { nodes, edges, canvasW, canvasH };
}
