import { CartesianGrid, Line, LineChart as Chart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { pct } from '../../chartFormat';
import { ChartLegend, TableView } from './ChartParts';

function LineTooltip({ active, payload, label, xLabel, formatX }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip">
      <div className="font-medium mb-1">{formatX ? formatX(label) : `${xLabel} ${label}`}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2 text-xs">
          <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: p.color }} aria-hidden />
          <span>{p.name}</span>
          <span className="ml-auto num">{pct(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * A line chart of shares (0 to 1) over steps.
 * data: [{ x, [seriesKey]: number }]; series: [{ key, name, color, dashed }].
 * marker: an optional x value to mark with a labelled vertical line.
 */
export default function LineChart({ data, series, xLabel, yLabel, height = 300, marker, markerLabel, formatX, title, yDomain }) {
  const values = data.flatMap((d) => series.map((s) => d[s.key])).filter((v) => typeof v === 'number');
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.05) * 10) / 10);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.05) * 10) / 10);
  const [from, to] = yDomain ?? [lo, hi];
  const step = to - from <= 0.3 ? 0.05 : to - from <= 0.6 ? 0.1 : 0.25;
  const ticks = Array.from({ length: Math.round((to - from) / step) + 1 }, (_, i) => Math.round((from + i * step) * 100) / 100);
  return (
    <figure className="m-0 flex flex-col gap-3" aria-label={title}>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <Chart data={data} margin={{ top: marker !== undefined ? 28 : 12, right: 24, bottom: 28, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="x" type="number" domain={['dataMin', 'dataMax']} allowDecimals={false}
              stroke="var(--axis)" tick={{ fill: 'var(--ink-2)', fontSize: 12 }} tickLine={false}
              tickFormatter={formatX} minTickGap={24}
              label={{ value: xLabel, position: 'insideBottom', offset: -16, fill: 'var(--ink-2)', fontSize: 12 }} />
            <YAxis domain={[from, to]} ticks={ticks} stroke="var(--axis)" tick={{ fill: 'var(--ink-2)', fontSize: 12 }} tickLine={false}
              axisLine={false} tickFormatter={pct} width={48}
              label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 8, fill: 'var(--ink-2)', fontSize: 12, style: { textAnchor: 'middle' } }} />
            {marker !== undefined && (
              <ReferenceLine x={marker} stroke="var(--reference)" strokeWidth={1}
                label={{ value: markerLabel, position: 'top', fill: 'var(--ink-2)', fontSize: 12 }} />
            )}
            <Tooltip content={<LineTooltip xLabel={xLabel} formatX={formatX} />} cursor={{ stroke: 'var(--axis)', strokeWidth: 1 }} />
            {series.map((s) => (
              <Line key={s.key} type="monotone" dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={2}
                strokeDasharray={s.dashed ? '6 4' : undefined} dot={false} isAnimationActive={false}
                activeDot={{ r: 5, stroke: 'var(--surface)', strokeWidth: 2 }} connectNulls />
            ))}
          </Chart>
        </ResponsiveContainer>
      </div>
      {series.length > 1 && <ChartLegend series={series} />}
      <TableView
        caption={title}
        columns={series.map((s) => s.name)}
        rows={data.map((d) => [formatX ? formatX(d.x) : `${xLabel} ${d.x}`, ...series.map((s) => pct(d[s.key]))])}
      />
    </figure>
  );
}
