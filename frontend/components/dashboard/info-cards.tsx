"use client"

import type React from "react"

import { useState } from "react"
import { Lightbulb, Target, CheckCircle, ArrowRight, ChevronDown, ChevronUp } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface InfoCardProps {
  title: string
  description: string
  icon: React.ReactNode
  color: string
  bullets: string[]
}

function InfoCard({ title, description, icon, color, bullets }: InfoCardProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card className={`${color} h-full transition-all duration-200`}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            {icon}
            <CardTitle>{title}</CardTitle>
          </div>
          <button onClick={() => setExpanded(!expanded)} className="p-1 rounded-full hover:bg-black/5">
            {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
        </div>
        <CardDescription className="text-gray-700">{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ul className={`space-y-2 transition-all duration-300 ${expanded ? "max-h-96" : "max-h-24 overflow-hidden"}`}>
          {bullets.map((bullet, index) => (
            <li key={index} className="flex items-start">
              <span className="mr-2 mt-1">•</span>
              <span>{bullet}</span>
            </li>
          ))}
        </ul>
        {!expanded && bullets.length > 2 && (
          <div className="text-right mt-2">
            <span className="text-sm text-gray-500 italic">{bullets.length - 2} more items...</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function InfoCards() {
  const cards = [
    {
      title: "Custom Solution",
      description: "Tailored ESG analysis framework",
      icon: <Lightbulb className="h-5 w-5" />,
      color: "bg-emerald-100 border-emerald-200",
      bullets: [
        "AI-powered ESG parameter analysis",
        "Sustainable Materials & Water tracking",
        "GHG Emissions & Energy monitoring",
        "Stakeholder Engagement metrics",
        "Social Responsibility reporting",
      ],
    },
    // Keep the rest of the cards the same
    {
      title: "Outcome of PoC",
      description: "Proof of concept results",
      icon: <CheckCircle className="h-5 w-5" />,
      color: "bg-green-50 border-green-100",
      bullets: [
        "90% accuracy in document extraction",
        "Reduced analysis time by 75%",
        "Identified 15 new improvement areas",
        "Standardized reporting across departments",
        "Positive feedback from stakeholders",
      ],
    },
    {
      title: "Target State",
      description: "Future implementation goals",
      icon: <Target className="h-5 w-5" />,
      color: "bg-blue-50 border-blue-100",
      bullets: [
        "Full integration with existing systems",
        "Automated monthly reporting",
        "Predictive ESG trend analysis",
        "Benchmarking against industry standards",
        "Mobile-friendly dashboard access",
      ],
    },
    {
      title: "Next Steps",
      description: "Implementation roadmap",
      icon: <ArrowRight className="h-5 w-5" />,
      color: "bg-purple-50 border-purple-100",
      bullets: [
        "Finalize data integration architecture",
        "Develop custom ML models for sector-specific analysis",
        "User acceptance testing with key stakeholders",
        "Training program development",
        "Phased rollout plan",
      ],
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => (
        <InfoCard key={index} {...card} />
      ))}
    </div>
  )
}
