import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

const ResultRadarChart = ({ data }) => {
    // Expected data: [{ subject: 'Skills', A: 80, fullMark: 100 }, ...]

    return (
        <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="60%" data={data}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar
                        name="Candidate"
                        dataKey="A"
                        stroke="#22d3ee" // Cyan-400
                        strokeWidth={2}
                        fill="#22d3ee"
                        fillOpacity={0.3}
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        itemStyle={{ color: '#22d3ee' }}
                    />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ResultRadarChart;
