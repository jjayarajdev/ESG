"use client"
import ESGPerformanceDashboard from "./esg-performance-dashboard"

interface AnalysisOutputProps {
  showFullReport?: boolean
}

export default function AnalysisOutput({ showFullReport = true }: AnalysisOutputProps) {
  return (
    <div className="space-y-6">
      <ESGPerformanceDashboard />
    </div>
  )
}
