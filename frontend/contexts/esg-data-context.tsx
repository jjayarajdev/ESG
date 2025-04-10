"use client"

import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"

// Define the data structure
export interface ESGDataItem {
  name: string
  value: number
  goal: number
  status: "green" | "red" | "grey"
  category?: string
  actual?: number
}

export interface ESGTableItem {
  category: string
  goal: string
  actual: string
  status: string
}

interface ESGDataContextType {
  esgData: ESGDataItem[]
  updateESGItem: (index: number, updates: Partial<ESGDataItem>) => void
  tableData: ESGTableItem[]
  updateTableItem: (index: number, updates: Partial<ESGTableItem>) => void
}

const ESGDataContext = createContext<ESGDataContextType | undefined>(undefined)

export function ESGDataProvider({ children }: { children: React.ReactNode }) {
  // Initial ESG data
  const [esgData, setESGData] = useState<ESGDataItem[]>([
    { name: "Sustainable Materials", value: 85, goal: 80, status: "green" },
    { name: "Water", value: 78, goal: 85, status: "red" },
    { name: "Energy", value: 92, goal: 80, status: "green" },
    { name: "Waste & Effluent", value: 76, goal: 80, status: "red" },
    { name: "Land Use/Animal Stewardship", value: 68, goal: 75, status: "red" },
    { name: "GHG Emissions", value: 83, goal: 80, status: "green" },
    { name: "Transportation", value: 70, goal: 80, status: "red" },
    { name: "Design & Operation", value: 88, goal: 85, status: "green" },
    { name: "Supply Chain Compliance", value: 65, goal: 80, status: "red" },
    { name: "Health & Wellbeing", value: 90, goal: 85, status: "green" },
    { name: "Inclusion", value: 82, goal: 85, status: "red" },
    { name: "Social Responsibility", value: 79, goal: 75, status: "green" },
    { name: "Stakeholder Engagement", value: 86, goal: 80, status: "green" },
  ])

  // Derived table data
  const [tableData, setTableData] = useState<ESGTableItem[]>([])

  // Function to update an ESG data item
  const updateESGItem = (index: number, updates: Partial<ESGDataItem>) => {
    const newData = [...esgData]
    newData[index] = { ...newData[index], ...updates }

    // Recalculate status if value or goal changed
    if (updates.value !== undefined || updates.goal !== undefined) {
      const value = updates.value !== undefined ? updates.value : newData[index].value
      const goal = updates.goal !== undefined ? updates.goal : newData[index].goal
      newData[index].status = value >= goal ? "green" : "red"
    }

    setESGData(newData)
  }

  // Function to update a table data item and sync with chart data
  const updateTableItem = (index: number, updates: Partial<ESGTableItem>) => {
    // Update the table data
    const newTableData = [...tableData]
    newTableData[index] = { ...newTableData[index], ...updates }
    setTableData(newTableData)

    // Sync with chart data
    const newESGData = [...esgData]

    // Handle goal updates
    if (updates.goal) {
      const goalValue = Number.parseInt(updates.goal.match(/\d+/)?.[0] || "0")
      newESGData[index].goal = goalValue
    }

    // Handle actual/value updates
    if (updates.actual) {
      const actualValue = Number.parseInt(updates.actual.match(/\d+/)?.[0] || "0")
      newESGData[index].value = actualValue
    }

    // Handle status updates
    if (updates.status) {
      newESGData[index].status = updates.status as "green" | "red" | "grey"
    }

    // Recalculate status if needed
    if (updates.actual || updates.goal) {
      newESGData[index].status = newESGData[index].value >= newESGData[index].goal ? "green" : "red"
    }

    setESGData(newESGData)
  }

  // Convert ESG data to table format
  useEffect(() => {
    const newTableData = esgData.map((item) => {
      // Convert numeric values to formatted strings with appropriate text
      let goalText = ""
      let actualText = ""

      // Format goal text based on the parameter
      switch (item.name) {
        case "Sustainable Materials":
          goalText = `${item.goal}% recycled content`
          actualText = `${item.value}% recycled content`
          break
        case "Water":
        case "Energy":
        case "GHG Emissions":
          goalText = `Reduce by ${item.goal}%`
          actualText = `Reduced by ${item.value}%`
          break
        case "Waste & Effluent":
          goalText = `${item.goal}% reduction`
          actualText = `${item.value}% reduction`
          break
        case "Land Use/Animal Stewardship":
          goalText = `Restore ${item.goal} acres`
          actualText = `${item.value} acres restored`
          break
        case "Transportation":
          goalText = `${item.goal}% electric fleet`
          actualText = `${item.value}% electric fleet`
          break
        case "Design & Operation":
          goalText = `${item.goal}% LEED certification`
          actualText = `${item.value}% LEED certified`
          break
        case "Supply Chain Compliance":
          goalText = `Audit ${item.goal}% of suppliers`
          actualText = `${item.value}% suppliers audited`
          break
        case "Health & Wellbeing":
          goalText = `${item.goal}% wellness program participation`
          actualText = `${item.value}% participation rate`
          break
        case "Inclusion":
          goalText = `${item.goal}% diverse workforce`
          actualText = `${item.value}% diverse workforce`
          break
        case "Social Responsibility":
          goalText = `${item.goal}% of profits to community`
          actualText = `${item.value}% of profits to community`
          break
        case "Stakeholder Engagement":
          goalText = `${item.goal}% engagement rate`
          actualText = `${item.value}% engagement achieved`
          break
        default:
          goalText = `${item.goal}% target`
          actualText = `${item.value}% achieved`
      }

      return {
        category: item.name,
        goal: goalText,
        actual: actualText,
        status: item.status,
      }
    })

    setTableData(newTableData)
  }, [esgData])

  return (
    <ESGDataContext.Provider value={{ esgData, updateESGItem, tableData, updateTableItem }}>
      {children}
    </ESGDataContext.Provider>
  )
}

export function useESGData() {
  const context = useContext(ESGDataContext)
  if (context === undefined) {
    throw new Error("useESGData must be used within an ESGDataProvider")
  }
  return context
}
