import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from 'recharts';
import { inWords, pct } from '../../chartFormat';
import { ChartLegend, TableView } from './ChartParts';

// Splits a long axis label over two lines so it never runs off the chart.
function AxisTick({ x, y, payload, textAnchor }) {
  const words = String(payload.value).split(' ');
  const lines = [];
  for (const word of words) {
    const last = lines[lines.length - 1];
    if (last && (last + ' ' + word).length <= 16) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return (
    <text x={x} y={y} textAnchor={textAnchor} fill="var(--ink-2)" fontSize={12}>
      {lines.map((line, i) => (
        <tspan key={line} x={x} dy={i === 0 ? (lines.length > 1 ? -6 : 4) : 14}>{line}</tspan>
      ))}
    </text>
  );
}

function SpiderTooltip({ active, payload, label, axes }) {
  if (!active || !payload?.length) return null;
  const axis = axes.find((a) => a.label === label);
  return (
    <div className="chart-tip" style={{ maxWidth: 260 }}>
      <div className="font-medium mb-1">{label}</div>
      {axis?.explain && <div className="muted text-xs mb-1.5">{axis.explain}</div>}
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 text-xs">
          <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: p.color }} aria-hidden />
          <span>{p.name}</span>
          <span className="ml-auto">{inWords(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * A spider (radar) chart of scores between 0 and 1.
 * axes: [{ key, label, explain }]; series: [{ key, name, color, values: { [axisKey]: number }, dashed }].
 */
export default function SpiderChart({ axes, series, height = 340, title }) {
  const data = axes.map((a) => ({ axis: a.label, ...Object.fromEntries(series.map((s) => [s.key, s.values[a.key] ?? 0])) }));
  return (
    <figure className="m-0 flex flex-col gap-3" aria-label={title}>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="68%" margin={{ top: 16, right: 40, bottom: 16, left: 40 }}>
            <PolarGrid stroke="var(--grid)" />
            <PolarAngleAxis dataKey="axis" tick={<AxisTick />} />
            <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} tickCount={5} />
            {series.map((s) => (
              <Radar key={s.key} name={s.name} dataKey={s.key} stroke={s.color} fill={s.color} fillOpacity={series.length > 1 ? 0.08 : 0.16}
                strokeWidth={2} strokeDasharray={s.dashed ? '6 4' : undefined} isAnimationActive={false}
                dot={{ r: 4, fill: s.color, stroke: 'var(--surface)', strokeWidth: 2 }} />
            ))}
            <Tooltip content={<SpiderTooltip axes={axes} />} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      {series.length > 1 && <ChartLegend series={series} />}
      <TableView
        caption={title}
        columns={series.map((s) => s.name)}
        rows={axes.map((a) => [a.label, ...series.map((s) => `${inWords(s.values[a.key])} (${pct(s.values[a.key])})`)])}
      />
    </figure>
  );
}
