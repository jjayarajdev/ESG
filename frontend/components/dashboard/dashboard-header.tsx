import { BarChart3 } from "lucide-react"

export default function DashboardHeader() {
  return (
    <div className="flex flex-col space-y-2">
      <div className="flex items-center space-x-2">
        <BarChart3 className="h-8 w-8 text-emerald-600" />
        <h1 className="text-3xl font-bold tracking-tight">ESG AI Analysis Platform</h1>
      </div>
      <p className="text-muted-foreground">Powered by LLM & ReactJS</p>
    </div>
  )
}
