"use client"
import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { useESGData } from "@/contexts/esg-data-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Edit, Save, X } from "lucide-react"

interface ESGTableProps {
  showFullReport?: boolean
  editable?: boolean
}

export default function ESGTable({ showFullReport = true, editable = false }: ESGTableProps) {
  const { tableData, updateTableItem } = useESGData()
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editValues, setEditValues] = useState<{ goal: string; actual: string }>({ goal: "", actual: "" })

  // Always display all items
  const displayData = tableData

  const handleEdit = (index: number) => {
    setEditingIndex(index)
    setEditValues({
      goal: displayData[index].goal,
      actual: displayData[index].actual,
    })
  }

  const handleSave = (index: number) => {
    updateTableItem(index, {
      goal: editValues.goal,
      actual: editValues.actual,
    })
    setEditingIndex(null)
  }

  const handleCancel = () => {
    setEditingIndex(null)
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[300px]">ESG Category</TableHead>
            <TableHead>Goal</TableHead>
            <TableHead>Actual</TableHead>
            <TableHead className="text-center">Status</TableHead>
            {editable && <TableHead className="w-[100px]">Actions</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {displayData.map((item, index) => (
            <TableRow key={item.category}>
              <TableCell className="font-medium">{item.category}</TableCell>
              <TableCell>
                {editingIndex === index ? (
                  <Input
                    value={editValues.goal}
                    onChange={(e) => setEditValues({ ...editValues, goal: e.target.value })}
                    className="w-full"
                  />
                ) : (
                  item.goal
                )}
              </TableCell>
              <TableCell>
                {editingIndex === index ? (
                  <Input
                    value={editValues.actual}
                    onChange={(e) => setEditValues({ ...editValues, actual: e.target.value })}
                    className="w-full"
                  />
                ) : (
                  item.actual
                )}
              </TableCell>
              <TableCell className="text-center">
                <StatusBadge status={item.status} />
              </TableCell>
              {editable && (
                <TableCell>
                  {editingIndex === index ? (
                    <div className="flex space-x-1">
                      <Button variant="ghost" size="icon" onClick={() => handleSave(index)}>
                        <Save className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={handleCancel}>
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ) : (
                    <Button variant="ghost" size="icon" onClick={() => handleEdit(index)}>
                      <Edit className="h-4 w-4" />
                    </Button>
                  )}
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const getStatusProps = (status: string) => {
    switch (status) {
      case "green":
        return {
          label: "On Target",
          className: "bg-green-100 text-green-800 hover:bg-green-100",
        }
      case "red":
        return {
          label: "Off Target",
          className: "bg-red-100 text-red-800 hover:bg-red-100",
        }
      case "grey":
        return {
          label: "No Data",
          className: "bg-gray-100 text-gray-800 hover:bg-gray-100",
        }
      default:
        return {
          label: "Unknown",
          className: "bg-gray-100 text-gray-800 hover:bg-gray-100",
        }
    }
  }

  const { label, className } = getStatusProps(status)

  return (
    <Badge variant="outline" className={className}>
      {label}
    </Badge>
  )
}
