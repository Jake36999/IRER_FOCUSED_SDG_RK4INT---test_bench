
import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

// Strict Prop Typing for Telemetry Data
export type TelemetryDataPoint = {
  timestamp: number;
  h_norm: number;
  rho_max: number;
  step: number;
};

interface TelemetryChartProps {
  data: TelemetryDataPoint[];
  loading?: boolean;
  width?: number;
  height?: number;
  onDataPointClick?: (t: number) => void;
  colorMode?: 'NORMAL' | 'CRITICAL';
}

// Pure component: only rerenders on prop change
export const TelemetryChart: React.FC<TelemetryChartProps> = React.memo(({
  data,
  loading = false,
  width = 600,
  height = 300,
  onDataPointClick,
  colorMode = 'NORMAL',
}) => {
  // ISA-101 Color Logic
  const strokeColor = colorMode === 'CRITICAL' ? '#ef4444' : '#3b82f6'; // Red vs Blue
  const rhoColor = colorMode === 'CRITICAL' ? '#f87171' : '#f59e0b'; // Red vs Amber
  const opacity = colorMode === 'CRITICAL' ? 1 : 0.85;

  // Memoize data for recharts
  const chartData = useMemo(() => data.map(d => ({ ...d })), [data]);

  // Custom Tooltip with ISA-101 Styling
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-xl text-xs font-mono">
          <p className="text-slate-400 mb-2">Step: {payload[0].payload.step}</p>
          <div className="flex items-center gap-2 text-blue-400">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            h_norm: {payload[0].value.toFixed(6)}
          </div>
          <div className="flex items-center gap-2 text-amber-400">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            rho_max: {payload[1].value.toFixed(4)}
          </div>
        </div>
      );
    }
    return null;
  };

  // Interactive click handler overlay
  const handleClick = (e: any) => {
    if (onDataPointClick && e && e.activePayload && e.activePayload[0]) {
      onDataPointClick(e.activePayload[0].payload.timestamp);
    }
  };

  return (
    <div className="w-full h-[300px] relative">
      {/* SVG Definitions for "Explosion" Glow Filters */}
      <svg height={0} width={0} className="absolute">
        <defs>
          <filter id="glow-blue" height="300%" width="300%" x="-75%" y="-75%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-amber" height="300%" width="300%" x="-75%" y="-75%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>

      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} onClick={handleClick} width={width} height={height}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis 
            dataKey="step" 
            stroke="#475569" 
            tick={{ fontSize: 10, fill: '#475569' }} 
            tickLine={false}
          />
          <YAxis 
            stroke="#475569" 
            tick={{ fontSize: 10, fill: '#475569' }} 
            tickLine={false}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#ffffff20' }} />
          {/* H_NORM Line - Primary Metric */}
          <Line
            type="monotone"
            dataKey="h_norm"
            stroke={strokeColor}
            strokeWidth={2}
            dot={false}
            filter="url(#glow-blue)"
            isAnimationActive={true}
            animationDuration={300}
            strokeOpacity={opacity}
          />
          {/* RHO_MAX Line - Secondary Metric (Warning Color) */}
          <Line
            type="monotone"
            dataKey="rho_max"
            stroke={rhoColor}
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 4"
            filter="url(#glow-amber)"
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Interactive Hit Layer for Data Points (optional, for custom click targets) */}
      {onDataPointClick && (
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} className="absolute top-0 left-0 pointer-events-none">
          {data.map((p, i) => (
            <rect
              key={p.timestamp}
              x={(i / data.length) * width - 5}
              y={0}
              width={10}
              height={height}
              fill="transparent"
              className="pointer-events-auto cursor-crosshair hover:bg-white/5"
              onClick={() => onDataPointClick(p.timestamp)}
            />
          ))}
        </svg>
      )}

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20 backdrop-blur-[1px]">
          <span className="text-cyan-500 text-xs animate-pulse">ACQUIRING STREAM...</span>
        </div>
      )}
    </div>
  );
});

TelemetryChart.displayName = 'TelemetryChart';