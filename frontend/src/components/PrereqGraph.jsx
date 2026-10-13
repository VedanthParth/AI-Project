import { useMemo, useState } from 'react';
import { Background, Controls, MarkerType, ReactFlow } from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import '@xyflow/react/dist/style.css';
import { pct } from '../api';

const NODE_W = 190;
const NODE_H = 54;

function layout(graph, pathSlots, candidates) {
  const step = new Map(pathSlots.map((slot, i) => [candidates[slot].id, i + 1]));
  const linked = new Set(graph.edges.flatMap((e) => [e.source, e.target]));
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'LR', nodesep: 14, ranksep: 70 });
  g.setDefaultEdgeLabel(() => ({}));
  graph.nodes.filter((n) => linked.has(n.id)).forEach((n) => g.setNode(String(n.id), { width: NODE_W, height: NODE_H }));
  graph.edges.forEach((e) => g.setEdge(String(e.source), String(e.target)));
  dagre.layout(g);

  // Unlinked nodes (path concepts with no inferred links) go in a row underneath, in path order.
  const positions = new Map();
  let bottom = 0;
  g.nodes().forEach((id) => {
    const pos = g.node(id);
    positions.set(id, { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 });
    bottom = Math.max(bottom, pos.y + NODE_H / 2);
  });
  graph.nodes
    .filter((n) => !linked.has(n.id))
    .sort((a, b) => (step.get(a.id) ?? 99) - (step.get(b.id) ?? 99))
    .forEach((n, i) => positions.set(String(n.id), { x: i * (NODE_W + 16), y: linked.size ? bottom + 40 : 0 }));

  const nodes = graph.nodes.map((n) => {
    const pick = step.get(n.id);
    const border = pick ? '2px solid var(--accent)' : n.candidate ? '1px solid var(--ink-2)' : '1px dashed var(--axis)';
    return {
      id: String(n.id),
      position: positions.get(String(n.id)),
      sourcePosition: 'right',
      targetPosition: 'left',
      draggable: false,
      connectable: false,
      data: {
        label: (
          <div className="text-left" title={n.name}>
            <div className="text-[11.5px] leading-tight font-medium" style={{
              display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
            }}>
              {pick ? `${pick}. ` : ''}{n.name}
            </div>
            <div className="text-[10.5px] muted mt-0.5">
              {n.seen ? `practised · mastery ${pct(n.mastery)}` : n.candidate ? 'candidate · not started' : 'prerequisite · not started'}
            </div>
          </div>
        ),
      },
      style: {
        width: NODE_W,
        height: NODE_H,
        padding: '6px 8px',
        borderRadius: 8,
        border,
        background: n.seen ? 'var(--surface-2)' : 'var(--surface)',
        color: 'var(--ink)',
        boxSizing: 'border-box',
      },
    };
  });
  const edges = graph.edges.map((e) => ({
    id: `${e.source}-${e.target}`,
    source: String(e.source),
    target: String(e.target),
    style: { stroke: 'var(--ink-2)', strokeWidth: 1.25 },
    markerEnd: { type: MarkerType.ArrowClosed, color: 'var(--ink-2)', width: 14, height: 14 },
  }));
  return { nodes, edges };
}

function visible(graph, pathIds, mode) {
  const linked = new Set(graph.edges.flatMap((e) => [e.source, e.target]));
  let keep;
  if (mode === 'path') {
    keep = new Set(pathIds);
    graph.edges.forEach((e) => {
      if (pathIds.has(e.target)) keep.add(e.source);
      if (pathIds.has(e.source)) keep.add(e.target);
    });
  } else {
    keep = new Set([...linked, ...pathIds]);
  }
  return {
    nodes: graph.nodes.filter((n) => keep.has(n.id)),
    edges: graph.edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
  };
}

export default function PrereqGraph({ example, graph, error, trainStudents }) {
  const [mode, setMode] = useState('path');
  const pathSlots = example.paths.pointer.slots;
  const flow = useMemo(() => {
    if (!graph) return null;
    const pathIds = new Set(pathSlots.map((slot) => example.candidates[slot].id));
    return layout(visible(graph, pathIds, mode), pathSlots, example.candidates);
  }, [graph, example, pathSlots, mode]);
  const linkedCandidates = graph
    ? graph.nodes.filter((n) => n.candidate && graph.edges.some((e) => e.source === n.id || e.target === n.id)).length
    : 0;

  return (
    <section className="card flex flex-col gap-3" aria-label="Prerequisite map">
      <div>
        <h2 className="card-title">Prerequisite map</h2>
        <p className="card-sub">
          Arrows run from a concept to one that training students almost always started after it (inferred from{' '}
          {trainStudents?.toLocaleString() ?? 'the training'} students' first attempts). Blue outlines are the
          recommended path; dashed boxes are prerequisites outside the candidate set.
        </p>
      </div>
      {error && <p className="bad m-0">Couldn't load the graph: {error}</p>}
      {!graph && !error && <div className="skeleton" style={{ height: 380 }} />}
      {flow && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex gap-1" role="group" aria-label="Graph scope">
              {[['path', 'Around the path'], ['all', 'All linked candidates']].map(([key, label]) => (
                <button key={key} type="button" className="btn text-xs" aria-pressed={mode === key} onClick={() => setMode(key)}
                  style={mode === key ? { borderColor: 'var(--accent)', background: 'var(--accent-soft)' } : undefined}>
                  {label}
                </button>
              ))}
            </div>
            <span className="muted text-xs num">
              {linkedCandidates} of {example.candidates.length} candidates have inferred prerequisite links
            </span>
          </div>
          <div style={{ height: mode === 'all' ? 480 : 340, border: '1px solid var(--border)', borderRadius: 8 }}>
            <ReactFlow key={mode} nodes={flow.nodes} edges={flow.edges} fitView fitViewOptions={{ maxZoom: 1.1, padding: 0.15 }}
              minZoom={0.2} nodesDraggable={false}
              nodesConnectable={false} proOptions={{ hideAttribution: true }} colorMode="system">
              <Background color="var(--grid)" gap={20} />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>
          {!flow.edges.length && (
            <p className="muted text-xs m-0">No inferred prerequisite links touch the recommended path here.</p>
          )}
        </>
      )}
    </section>
  );
}
