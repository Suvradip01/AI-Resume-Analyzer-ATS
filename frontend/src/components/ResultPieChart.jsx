import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { useState } from 'react';

const ResultPieChart = ({ score }) => {
    const data = [
        { name: 'Score', value: score, color: '#8b5cf6' }, // Violet-500
        { name: 'Remaining', value: 100 - score, color: '#1e293b' }, // Slate-800
    ];

    const [hoveredData, setHoveredData] = useState(null);

    return (
        <div className="relative h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={80}
                        outerRadius={110}
                        startAngle={90}
                        endAngle={-270}
                        dataKey="value"
                        stroke="none"
                        onMouseEnter={(_, index) => setHoveredData(data[index])}
                        onMouseLeave={() => setHoveredData(null)}
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                    </Pie>
                    <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                        itemStyle={{ color: '#8b5cf6' }}
                        formatter={(value) => `${Math.round(value)}%`}
                    />
                </PieChart>
            </ResponsiveContainer>

            {/* Center Label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-4xl font-bold text-white">
                    {Math.round(score)}
                </span>
                <span className="text-sm text-gray-400">Total Score</span>
            </div>
        </div>
    );
};

export default ResultPieChart;
