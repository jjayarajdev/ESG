import type React from "react"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts"

interface CartwheelChartProps {
  data: {
    category: string
    goal: number
    actual: number
    status?: "green" | "red" | "grey"
  }[]
}

const CartwheelChart: React.FC<CartwheelChartProps> = ({ data }) => {
  // Create separate datasets for each status
  const processedData = data.map((item) => ({
    category: item.category,
    goal: item.goal,
    greenValue: item.status === "green" ? item.actual : 0,
    redValue: item.status === "red" ? item.actual : 0,
    greyValue: item.status === "grey" ? item.actual : 0,
    actual: item.actual,
  }))

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={processedData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="category" />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />

          {/* Goal line (dashed) */}
          <Radar name="Goal" dataKey="goal" stroke="#888888" fill="none" strokeWidth={1.5} strokeDasharray="5 5" />

          {/* Green segments - On Target */}
          <Radar
            name="On Target"
            dataKey="greenValue"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.6}
            strokeWidth={2}
          />

          {/* Red segments - Off Target */}
          <Radar
            name="Off Target"
            dataKey="redValue"
            stroke="#ef4444"
            fill="#ef4444"
            fillOpacity={0.6}
            strokeWidth={2}
          />

          {/* Grey segments - No Data */}
          <Radar name="No Data" dataKey="greyValue" stroke="#9ca3af" fill="#9ca3af" fillOpacity={0.6} strokeWidth={2} />

          <Tooltip
            formatter={(value, name, props) => {
              if (value === 0) return [null, null] // Don't show zero values
              if (name === "Goal") return [`Goal: ${value}%`, name]

              const item = props.payload
              return [`Actual: ${item.actual}%, Goal: ${item.goal}%`, name]
            }}
          />

          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default CartwheelChart
