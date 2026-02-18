import { useState } from 'react';
import { Group } from '@visx/group';
import { scaleLinear } from '@visx/scale';
import { motion, AnimatePresence } from 'framer-motion';

const ResultRadarChart = ({ data }) => {
    // data: [{ subject: 'Skills', A: 80, fullMark: 100 }, ...]
    const [hoveredIndex, setHoveredIndex] = useState(null);

    // ── Configuration ──────────────────────────────────────────────
    const width = 310; // Reduced to give more room to legend
    const height = 280;
    const margin = { top: 40, right: 20, bottom: 40, left: 70 };

    // Chart center is the middle of the drawing area
    const centerX = (width - margin.left - margin.right) / 2;
    const centerY = (height - margin.top - margin.bottom) / 2;
    const maxRadius = Math.min(centerX, centerY) - 15;

    // Color mapping for each metric
    const getColorForMetric = (subject) => {
        const colors = {
            'Skills': '#3b82f6',      // Blue
            'Experience': '#10b981',   // Green
            'Projects': '#f59e0b',     // Amber/Orange
            'Structure': '#8b5cf6'     // Violet/Purple
        };
        return colors[subject] || '#22d3ee';
    };

    const numAxes = data.length;
    const angleSlice = (Math.PI * 2) / numAxes;

    const radiusScale = scaleLinear({
        domain: [0, 100],
        range: [0, maxRadius],
    });

    const getAdjustedValue = (val) => (val > 0 ? Math.max(16, val) : 0);

    const gridLevels = [20, 40, 60, 80, 100];

    // ── Helpers ────────────────────────────────────────────────────
    const getPoint = (index, value) => {
        const angle = angleSlice * index - Math.PI / 2;
        const r = radiusScale(value);
        return {
            x: centerX + r * Math.cos(angle),
            y: centerY + r * Math.sin(angle),
        };
    };

    // Generate polygon for a specific metric series (irregular shape)
    const generateSeriesPolygon = (seriesIndex) => {
        const baseValue = data[seriesIndex].A;
        const primaryValue = getAdjustedValue(baseValue);

        return data.map((_, i) => {
            const angle = angleSlice * i - Math.PI / 2;

            // Primary axis gets the value
            // Neighbors get a fixed fraction of the PRIMARY value to be symmetric
            let radiusValue = 2;

            if (i === seriesIndex) {
                radiusValue = primaryValue;
            } else if (i === (seriesIndex + 1) % numAxes || i === (seriesIndex + numAxes - 1) % numAxes) {
                // Symmetric "wings" - increased to 0.65 for more fill
                radiusValue = primaryValue * 0.65;
            }

            const r = radiusScale(radiusValue);
            const x = centerX + r * Math.cos(angle);
            const y = centerY + r * Math.sin(angle);
            return `${i === 0 ? 'M' : 'L'} ${x},${y}`;
        }).join(' ') + ' Z';
    };

    const axisEndpoints = data.map((_, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        return {
            x2: centerX + maxRadius * Math.cos(angle),
            y2: centerY + maxRadius * Math.sin(angle),
        };
    });

    const axisLabels = data.map((item, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const labelRadius = maxRadius + 38; // Slightly more spacing for larger chart
        return {
            x: centerX + labelRadius * Math.cos(angle),
            y: centerY + labelRadius * Math.sin(angle),
            text: item.subject,
        };
    });

    const dataPoints = data.map((item, i) => ({
        ...getPoint(i, getAdjustedValue(item.A)), // Align dots with adjusted polygon tips
        value: item.A,
        subject: item.subject,
    }));

    // ── Render ─────────────────────────────────────────────────────
    return (
        <div className="w-full flex flex-col xl:flex-row items-center justify-center gap-x-6 gap-y-10 px-2">

            {/* ── Chart ── */}
            <div className="flex-shrink-0">
                <svg width={width} height={height}>
                    <Group left={margin.left} top={margin.top}>

                        {/* Grid Polygons (Diamond) */}
                        {gridLevels.map((level, i) => {
                            const points = data.map((_, idx) => {
                                const angle = angleSlice * idx - Math.PI / 2;
                                const r = radiusScale(level);
                                return `${centerX + r * Math.cos(angle)},${centerY + r * Math.sin(angle)}`;
                            }).join(' ');

                            return (
                                <polygon
                                    key={`grid-${i}`}
                                    points={points}
                                    fill="none"
                                    stroke="#334155"
                                    strokeWidth={0.75}
                                    opacity={0.4}
                                />
                            );
                        })}

                        {/* Axis lines */}
                        {axisEndpoints.map((pt, i) => (
                            <line
                                key={`axis-${i}`}
                                x1={centerX} y1={centerY}
                                x2={pt.x2} y2={pt.y2}
                                stroke="#334155"
                                strokeWidth={0.75}
                                opacity={0.5}
                            />
                        ))}

                        {/* Multiple overlapping polygons - one for each metric */}
                        {data.map((seriesItem, seriesIndex) => {
                            const isHovered = hoveredIndex === seriesIndex;
                            const isFaded = hoveredIndex !== null && hoveredIndex !== seriesIndex;
                            const color = getColorForMetric(seriesItem.subject);
                            const pathData = generateSeriesPolygon(seriesIndex);

                            return (
                                <g
                                    key={`series-${seriesIndex}`}
                                    onMouseEnter={() => setHoveredIndex(seriesIndex)}
                                    onMouseLeave={() => setHoveredIndex(null)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    {/* Glow effect when hovered */}
                                    <AnimatePresence>
                                        {isHovered && (
                                            <motion.path
                                                d={pathData}
                                                fill="none"
                                                stroke={color}
                                                strokeWidth={10}
                                                style={{ filter: 'blur(10px)' }}
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 0.4 }}
                                                exit={{ opacity: 0 }}
                                                transition={{ duration: 0.2 }}
                                            />
                                        )}
                                    </AnimatePresence>

                                    {/* Main polygon for this series */}
                                    <motion.path
                                        d={pathData}
                                        fill={color}
                                        fillOpacity={isHovered ? 0.35 : isFaded ? 0.08 : 0.2}
                                        stroke={color}
                                        strokeWidth={isHovered ? 2.5 : 1.5}
                                        initial={{ opacity: 0, scale: 0 }}
                                        animate={{
                                            opacity: 1,
                                            scale: isHovered ? 1.03 : 1,
                                            fillOpacity: isHovered ? 0.35 : isFaded ? 0.08 : 0.2,
                                            strokeWidth: isHovered ? 2.5 : 1.5
                                        }}
                                        transition={{
                                            duration: 0.3,
                                            delay: seriesIndex * 0.08,
                                            scale: { duration: 0.2 }
                                        }}
                                        style={{ transformOrigin: `${centerX}px ${centerY}px` }}
                                    />
                                </g>
                            );
                        })}


                        {/* Data points */}
                        {dataPoints.map((point, i) => {
                            const isHovered = hoveredIndex === i;
                            const color = getColorForMetric(point.subject);
                            return (
                                <g key={`point-${i}`}>
                                    {/* Hover glow ring */}
                                    <AnimatePresence>
                                        {isHovered && (
                                            <motion.circle
                                                cx={point.x}
                                                cy={point.y}
                                                r={14}
                                                fill={color}
                                                opacity={0.15}
                                                initial={{ scale: 0, opacity: 0 }}
                                                animate={{ scale: 1, opacity: 0.15 }}
                                                exit={{ scale: 0, opacity: 0 }}
                                                transition={{ duration: 0.15 }}
                                            />
                                        )}
                                    </AnimatePresence>

                                    {/* Point itself */}
                                    <motion.circle
                                        cx={point.x}
                                        cy={point.y}
                                        r={point.value > 0 ? 4 : 2}
                                        fill={color}
                                        stroke="white"
                                        strokeWidth={1.5}
                                        initial={{ opacity: 0, scale: 0 }}
                                        animate={{
                                            opacity: 1,
                                            scale: isHovered ? 1.6 : 1,
                                        }}
                                        transition={{
                                            opacity: { delay: i * 0.08 + 0.3 },
                                            scale: { duration: 0.15 },
                                        }}
                                        onMouseEnter={() => setHoveredIndex(i)}
                                        onMouseLeave={() => setHoveredIndex(null)}
                                        style={{ cursor: 'pointer', transformOrigin: `${point.x}px ${point.y}px` }}
                                    />

                                    {/* Tooltip on hover */}
                                    <AnimatePresence>
                                        {isHovered && (
                                            <motion.g
                                                initial={{ opacity: 0, y: -4 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: -4 }}
                                                transition={{ duration: 0.15 }}
                                            >
                                                <rect
                                                    x={point.x - 22}
                                                    y={point.y - 28}
                                                    width={44}
                                                    height={20}
                                                    rx={6}
                                                    fill="#1e293b"
                                                    stroke={color}
                                                    strokeWidth={1}
                                                />
                                                <text
                                                    x={point.x}
                                                    y={point.y - 14}
                                                    textAnchor="middle"
                                                    dominantBaseline="middle"
                                                    fontSize={11}
                                                    fontWeight={700}
                                                    fill={color}
                                                    fontFamily="monospace"
                                                >
                                                    {point.value}%
                                                </text>
                                            </motion.g>
                                        )}
                                    </AnimatePresence>
                                </g>
                            );
                        })}

                        {/* Axis labels */}
                        {axisLabels.map((label, i) => {
                            const isHovered = hoveredIndex === i;
                            return (
                                <text
                                    key={`label-${i}`}
                                    x={label.x}
                                    y={label.y}
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                    fontSize={11}
                                    fontWeight={isHovered ? 700 : 500}
                                    fill={isHovered ? '#ffffff' : '#94a3b8'}
                                    style={{ userSelect: 'none', transition: 'fill 0.2s, font-weight 0.2s' }}
                                >
                                    {label.text}
                                </text>
                            );
                        })}
                    </Group>
                </svg>
            </div>

            {/* ── Legend ── */}
            <div className="flex flex-col gap-1.5 min-w-[140px] flex-1 max-w-[200px]">
                <h3 className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">
                    Performance Breakdown
                </h3>

                {data.map((item, index) => {
                    const isHovered = hoveredIndex === index;
                    const isAny = hoveredIndex !== null;
                    const color = getColorForMetric(item.subject);

                    return (
                        <motion.div
                            key={`legend-${index}`}
                            className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg cursor-pointer"
                            style={{
                                background: isHovered ? `${color}15` : 'transparent',
                                border: `1px solid ${isHovered ? `${color}55` : 'rgba(255,255,255,0.04)'}`,
                                opacity: isAny && !isHovered ? 0.35 : 1,
                                transition: 'background 0.2s, border 0.2s, opacity 0.2s',
                            }}
                            onMouseEnter={() => setHoveredIndex(index)}
                            onMouseLeave={() => setHoveredIndex(null)}
                            initial={{ opacity: 0, x: 16 }}
                            animate={{ opacity: isAny && !isHovered ? 0.35 : 1, x: 0 }}
                            transition={{ delay: index * 0.07 }}
                        >
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                                    style={{
                                        backgroundColor: color,
                                        boxShadow: isHovered ? `0 0 8px ${color}` : 'none',
                                        transition: 'box-shadow 0.2s',
                                    }}
                                />
                                <span className="text-sm text-slate-300 font-medium">
                                    {item.subject}
                                </span>
                            </div>
                            <span
                                className="text-sm font-bold tabular-nums"
                                style={{ color: isHovered ? color : '#94a3b8', transition: 'color 0.2s' }}
                            >
                                {Math.round(item.A)}%
                            </span>
                        </motion.div>
                    );
                })}
            </div>
        </div>
    );
};

export default ResultRadarChart;