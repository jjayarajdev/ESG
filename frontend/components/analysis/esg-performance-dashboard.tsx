"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PieChart, TableIcon, Download, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ESGDataProvider } from "@/contexts/esg-data-context"
import ESGChart from "./esg-chart"
import ESGTable from "./esg-table"

export default function ESGPerformanceDashboard() {
  const [viewType, setViewType] = useState<"chart" | "table">("chart")
  const [editMode, setEditMode] = useState(false)

  const handleExport = () => {
    alert("Exporting to Excel...")
    // Implementation for Excel export would go here
  }

  const toggleEditMode = () => {
    setEditMode(!editMode)
    // If switching to chart view while in edit mode, switch back to table view
    if (!editMode && viewType === "chart") {
      setViewType("table")
    }
  }

  return (
    <ESGDataProvider>
      <Card className="w-full">
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            <div>
              <CardTitle>ESG Performance Dashboard</CardTitle>
              <CardDescription>Comprehensive analysis of ESG performance across all 13 parameters</CardDescription>
            </div>
            <div className="flex space-x-2">
              <Tabs value={viewType} onValueChange={(v) => setViewType(v as "chart" | "table")} className="w-[200px]">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="chart" disabled={editMode}>
                    <PieChart className="h-4 w-4 mr-2" />
                    Chart
                  </TabsTrigger>
                  <TabsTrigger value="table">
                    <TableIcon className="h-4 w-4 mr-2" />
                    Table
                  </TabsTrigger>
                </TabsList>
              </Tabs>
              <Button variant="outline" size="icon" onClick={handleExport} title="Export to Excel">
                <Download className="h-4 w-4" />
              </Button>
              <Button
                variant={editMode ? "default" : "outline"}
                size="icon"
                onClick={toggleEditMode}
                title={editMode ? "Save Changes" : "Edit Data"}
              >
                <Edit className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {viewType === "chart" ? <ESGChart /> : <ESGTable showFullReport={true} editable={editMode} />}
        </CardContent>
      </Card>
    </ESGDataProvider>
  )
}
