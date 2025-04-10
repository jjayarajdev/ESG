"use client"

import { useState } from "react"
import DashboardHeader from "./dashboard-header"
import DocumentIngestion from "../document/document-ingestion"
import AnalysisOutput from "../analysis/analysis-output"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("documents")

  return (
    <div className="container mx-auto px-4 py-8">
      <DashboardHeader />

      <Tabs defaultValue="documents" className="mt-6" onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2 mb-8">
          <TabsTrigger value="documents">Document Analysis</TabsTrigger>
          <TabsTrigger value="reports">ESG Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="documents" className="p-0">
          <DocumentIngestion />
        </TabsContent>

        <TabsContent value="reports">
          <AnalysisOutput showFullReport={true} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
