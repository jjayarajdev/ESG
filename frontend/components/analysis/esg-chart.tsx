"use client"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useESGData } from "@/contexts/esg-data-context"

export default function ESGChart() {
  // Use the shared context
  const { esgData, updateESGItem } = useESGData()

  // Create separate datasets for each status
  const greenData = esgData.map((item) => ({
    ...item,
    greenValue: item.status === "green" ? item.value : 0,
  }))

  const redData = esgData.map((item) => ({
    ...item,
    redValue: item.status === "red" ? item.value : 0,
  }))

  const greyData = esgData.map((item) => ({
    ...item,
    greyValue: item.status === "grey" ? item.value : 0,
  }))

  // Combine all data
  const combinedData = esgData.map((item, index) => ({
    name: item.name,
    goal: item.goal,
    greenValue: greenData[index].greenValue,
    redValue: redData[index].redValue,
    greyValue: greyData[index].greyValue,
    status: item.status,
    value: item.value,
    index, // Keep track of the original index
  }))

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>ESG Performance Analysis</CardTitle>
        <CardDescription>Performance across all 13 ESG parameters relative to goals</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={combinedData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" />
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
              <Radar
                name="No Data"
                dataKey="greyValue"
                stroke="#9ca3af"
                fill="#9ca3af"
                fillOpacity={0.6}
                strokeWidth={2}
              />

              <Tooltip
                formatter={(value, name, props) => {
                  if (value === 0) return [null, null] // Don't show zero values
                  if (name === "Goal") return [`Goal: ${value}%`, name]

                  const item = props.payload
                  return [`Actual: ${item.value}%, Goal: ${item.goal}%`, name]
                }}
                labelFormatter={(label) => `${label}`}
              />

              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-center text-sm text-gray-500">
          Green segments indicate targets achieved, red segments indicate improvement areas
        </div>
      </CardContent>
    </Card>
  )
}
