import { Pie } from '@visx/shape';
import { Group } from '@visx/group';
import { scaleOrdinal } from '@visx/scale';
import { useState } from 'react';
import { motion } from 'framer-motion';

const ResultPieChart = ({ score }) => {
    const data = [
        { label: 'Score', value: score, color: '#8b5cf6' }, // Violet-500
        { label: 'Remaining', value: 100 - score, color: '#1e293b' }, // Slate-800
    ];

    const [hoveredIndex, setHoveredIndex] = useState(null);
    const [tooltipData, setTooltipData] = useState(null);

    const width = 260; // Slightly increased
    const height = 260;
    const centerX = width / 2;
    const centerY = height / 2;
    const innerRadius = 65;
    const outerRadius = 105;

    const colorScale = scaleOrdinal({
        domain: data.map(d => d.label),
        range: data.map(d => d.color),
    });

    return (
        <div className="relative h-64 w-full flex items-center justify-center">
            <svg width={width} height={height}>
                <Group top={centerY} left={centerX}>
                    <Pie
                        data={data}
                        pieValue={d => d.value}
                        outerRadius={outerRadius}
                        innerRadius={innerRadius}
                        cornerRadius={0}
                        padAngle={0.02}
                    >
                        {pie => {
                            return pie.arcs.map((arc, index) => {
                                const isHovered = hoveredIndex === index;
                                const arcPath = pie.path(arc);

                                return (
                                    <g
                                        key={`arc-${index}`}
                                        onMouseEnter={() => {
                                            setHoveredIndex(index);
                                            setTooltipData({ label: arc.data.label, value: arc.data.value, percentage: ((arc.data.value / (score + (100 - score))) * 100).toFixed(1) });
                                        }}
                                        onMouseLeave={() => {
                                            setHoveredIndex(null);
                                            setTooltipData(null);
                                        }}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        <motion.path
                                            d={arcPath}
                                            fill={colorScale(arc.data.label)}
                                            initial={{ opacity: 0, scale: 0 }}
                                            animate={{
                                                opacity: isHovered && index === 0 ? 0.9 : hoveredIndex !== null && hoveredIndex !== index ? 0.5 : 1,
                                                scale: isHovered ? 1.05 : 1,
                                            }}
                                            transition={{ duration: 0.2 }}
                                        />
                                        {isHovered && index === 0 && (
                                            <motion.path
                                                d={arcPath}
                                                fill="url(#glow)"
                                                initial={{ opacity: 0 }}
                                                animate={{ opacity: 0.3 }}
                                                style={{ filter: 'blur(8px)' }}
                                            />
                                        )}
                                    </g>
                                );
                            });
                        }}
                    </Pie>
                    <defs>
                        <radialGradient id="glow">
                            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="1" />
                            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
                        </radialGradient>
                    </defs>
                </Group>
            </svg>

            {/* PieCenter - Center Label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <motion.span
                    className="text-4xl font-bold text-white"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    {Math.round(score)}
                </motion.span>
                <motion.span
                    className="text-sm text-gray-400"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                >
                    Total Score
                </motion.span>
            </div>

            {/* Tooltip */}
            {tooltipData && (
                <motion.div
                    className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full mb-2 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 shadow-xl"
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.15 }}
                >
                    <div className="text-sm text-white font-medium">{tooltipData.label}</div>
                    <div className="text-xs text-violet-400 font-bold">{tooltipData.percentage}%</div>
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-full w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-slate-700"></div>
                </motion.div>
            )}
        </div>
    );
};

export default ResultPieChart;

