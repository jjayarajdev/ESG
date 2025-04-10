"use client"

import { useState, useRef, useEffect } from "react"
import type React from "react"

import { PaperclipIcon, SendIcon, Mic, Plus, MessageSquare, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  attachments?: string[]
}

interface ChatSession {
  id: string
  title: string
  lastMessage: Date
}

export default function DocumentIngestion() {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([
    { id: "1", title: "Carbon Emissions Analysis", lastMessage: new Date(Date.now() - 3600000) },
    { id: "2", title: "Diversity Metrics Review", lastMessage: new Date(Date.now() - 86400000) },
    { id: "3", title: "Governance Policy Evaluation", lastMessage: new Date(Date.now() - 172800000) },
  ])
  const [currentSession, setCurrentSession] = useState<string | null>(null)
  const [attachments, setAttachments] = useState<string[]>([])
  const [files, setFiles] = useState<File[]>([])
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [citations, setCitations] = useState<string[]>([])
  const [isValidated, setIsValidated] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [showEstimatePrompt, setShowEstimatePrompt] = useState(false)
  const [isEstimated, setIsEstimated] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files)
      setFiles((prev) => [...prev, ...fileArray])
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleAskQuestion = () => {
    if (!question.trim() || files.length === 0) return

    setIsLoading(true)
    setAnswer("")
    setCitations([])
    setIsValidated(false)
    setIsEstimated(false)

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false)

      // Simulate vague answer detection
      const isVague = Math.random() > 0.5

      if (isVague) {
        setAnswer(
          "Based on the documents, there appears to be some information related to your query, but the exact details are not clearly specified.",
        )
        setCitations(["Document 1, page 23", "Document 2, page 45"])
        setShowEstimatePrompt(true)
      } else {
        setAnswer(
          "The company has reduced carbon emissions by 15% year-over-year through implementation of energy efficiency measures and renewable energy procurement. This exceeds their target of 10% reduction.",
        )
        setCitations(["ESG Report 2023, page 12", "Sustainability Strategy Document, page 8"])
        setShowEstimatePrompt(false)
      }
    }, 2000)
  }

  const handleEstimateAnswer = (shouldEstimate: boolean) => {
    setShowEstimatePrompt(false)

    if (shouldEstimate) {
      setIsEstimated(true)
      setAnswer(
        "Based on the available information and industry benchmarks, we estimate that the company has achieved approximately 12-14% reduction in carbon emissions, which would meet their stated goals.",
      )
    }
  }

  const handleSend = () => {
    if (!input.trim() && attachments.length === 0) return

    // Create a new session if none is selected
    if (!currentSession) {
      const newSessionId = Date.now().toString()
      const newSession = {
        id: newSessionId,
        title: input.slice(0, 30) + (input.length > 30 ? "..." : ""),
        lastMessage: new Date(),
      }
      setChatSessions((prev) => [newSession, ...prev])
      setCurrentSession(newSessionId)
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
      attachments: attachments.length > 0 ? [...attachments] : undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setAttachments([])
    setIsLoading(true)

    // Simulate AI response after a delay
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: generateResponse(input),
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, aiResponse])
      setIsLoading(false)

      // Update session last message time
      if (currentSession) {
        setChatSessions((prev) =>
          prev.map((session) => (session.id === currentSession ? { ...session, lastMessage: new Date() } : session)),
        )
      }
    }, 1500)
  }

  const generateResponse = (query: string): string => {
    // Simple response generation logic
    if (query.toLowerCase().includes("carbon")) {
      return "Based on the uploaded documents, your company has reduced carbon emissions by 15% year-over-year through implementation of energy efficiency measures and renewable energy procurement. This exceeds the target of 10% reduction as stated in your sustainability report (page 12)."
    } else if (query.toLowerCase().includes("diversity")) {
      return "The diversity metrics show 38% women in leadership positions, which is slightly below your target of 40%. However, this represents a 5% improvement from the previous year. The documents indicate several initiatives in place to improve this metric further in the coming year."
    } else {
      return "I've analyzed your ESG documents and found relevant information related to your query. Would you like me to provide specific metrics, recommendations, or extract particular sections from the documents?"
    }
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files)
      const newAttachments = fileArray.map((file) => file.name)
      setAttachments((prev) => [...prev, ...newAttachments])
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const startNewChat = () => {
    setCurrentSession(null)
    setMessages([])
    setInput("")
    setAttachments([])
  }

  const selectSession = (sessionId: string) => {
    setCurrentSession(sessionId)
    // In a real app, you would load messages for this session
    // For now, we'll just simulate different messages
    const sampleMessages: Message[] = [
      {
        id: "m1",
        role: "user",
        content:
          sessionId === "1"
            ? "What are our carbon emission metrics for this year?"
            : sessionId === "2"
              ? "How are we doing on diversity targets?"
              : "Can you analyze our governance policies?",
        timestamp: new Date(Date.now() - 3600000),
      },
      {
        id: "m2",
        role: "assistant",
        content: generateResponse(sessionId === "1" ? "carbon" : sessionId === "2" ? "diversity" : "governance"),
        timestamp: new Date(Date.now() - 3500000),
      },
    ]
    setMessages(sampleMessages)
  }

  const deleteSession = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    setChatSessions((prev) => prev.filter((session) => session.id !== sessionId))
    if (currentSession === sessionId) {
      setCurrentSession(null)
      setMessages([])
    }
  }

  const formatDate = (date: Date) => {
    const today = new Date()
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)

    if (date.toDateString() === today.toDateString()) {
      return "Today"
    } else if (date.toDateString() === yesterday.toDateString()) {
      return "Yesterday"
    } else {
      return date.toLocaleDateString()
    }
  }

  return (
    <div className="flex h-[calc(100vh-12rem)] overflow-hidden rounded-lg border">
      {/* Left sidebar - Chat history */}
      <div className="w-64 border-r bg-gray-50 flex flex-col">
        <div className="p-3 border-b">
          <Button variant="outline" className="w-full justify-start text-left font-normal" onClick={startNewChat}>
            <Plus className="mr-2 h-4 w-4" />
            New chat
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-2">
            {chatSessions.map((session) => (
              <div
                key={session.id}
                className={cn(
                  "flex items-center justify-between p-2 rounded-md cursor-pointer hover:bg-gray-100 group",
                  currentSession === session.id && "bg-gray-100",
                )}
                onClick={() => selectSession(session.id)}
              >
                <div className="flex items-center space-x-2 truncate">
                  <MessageSquare className="h-4 w-4 shrink-0 text-gray-500" />
                  <span className="truncate text-sm">{session.title}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 opacity-0 group-hover:opacity-100"
                  onClick={(e) => deleteSession(e, session.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {/* Messages area */}
        <ScrollArea className="flex-1 p-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-8">
              <h2 className="text-2xl font-semibold mb-2">ESG Document Analysis</h2>
              <p className="text-muted-foreground mb-6 max-w-md">
                Upload ESG documents and ask questions to get AI-powered insights and analysis.
              </p>
              <div className="grid grid-cols-2 gap-3 w-full max-w-md">
                <Button
                  variant="outline"
                  className="justify-start text-left h-auto p-3"
                  onClick={() => setInput("What are our carbon emission targets?")}
                >
                  <div>
                    <p className="font-medium">Carbon emissions</p>
                    <p className="text-xs text-muted-foreground">Analyze carbon reduction progress</p>
                  </div>
                </Button>
                <Button
                  variant="outline"
                  className="justify-start text-left h-auto p-3"
                  onClick={() => setInput("How diverse is our leadership team?")}
                >
                  <div>
                    <p className="font-medium">Diversity metrics</p>
                    <p className="text-xs text-muted-foreground">Review diversity & inclusion data</p>
                  </div>
                </Button>
                <Button
                  variant="outline"
                  className="justify-start text-left h-auto p-3"
                  onClick={() => setInput("Summarize our governance policies")}
                >
                  <div>
                    <p className="font-medium">Governance</p>
                    <p className="text-xs text-muted-foreground">Evaluate governance structure</p>
                  </div>
                </Button>
                <Button
                  variant="outline"
                  className="justify-start text-left h-auto p-3"
                  onClick={() => setInput("What are our sustainability goals?")}
                >
                  <div>
                    <p className="font-medium">Sustainability</p>
                    <p className="text-xs text-muted-foreground">Review sustainability initiatives</p>
                  </div>
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <div key={message.id} className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}>
                  <div
                    className={cn(
                      "max-w-[80%] rounded-lg p-4",
                      message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted",
                    )}
                  >
                    {message.attachments && message.attachments.length > 0 && (
                      <div className="mb-2 space-y-1">
                        {message.attachments.map((file, index) => (
                          <div key={index} className="text-xs flex items-center">
                            <PaperclipIcon className="h-3 w-3 mr-1" />
                            <span>{file}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    <div
                      className={cn(
                        "text-xs mt-1",
                        message.role === "user" ? "text-primary-foreground/70" : "text-muted-foreground",
                      )}
                    >
                      {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted max-w-[80%] rounded-lg p-4">
                    <div className="flex space-x-2">
                      <div
                        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      ></div>
                      <div
                        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      ></div>
                      <div
                        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input area */}
        <div className="p-4 border-t">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div key={index} className="bg-muted text-xs py-1 px-2 rounded-full flex items-center">
                  <span className="truncate max-w-[150px]">{file}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-4 w-4 ml-1"
                    onClick={() => setAttachments((prev) => prev.filter((_, i) => i !== index))}
                  >
                    <span>×</span>
                  </Button>
                </div>
              ))}
            </div>
          )}
          <div className="relative">
            <Textarea
              placeholder="Ask about your ESG documents..."
              className="min-h-[60px] resize-none pr-20 py-3"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <div className="absolute right-2 bottom-2 flex items-center space-x-1">
              <input type="file" ref={fileInputRef} className="hidden" multiple onChange={handleFileUpload} />
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full"
                onClick={() => fileInputRef.current?.click()}
              >
                <PaperclipIcon className="h-5 w-5" />
              </Button>
              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
                <Mic className="h-5 w-5" />
              </Button>
              <Button
                size="icon"
                className="h-8 w-8 rounded-full"
                onClick={handleSend}
                disabled={!input.trim() && attachments.length === 0}
              >
                <SendIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <div className="mt-2 text-xs text-center text-muted-foreground">
            ESG AI may produce inaccurate information about documents or metrics.
          </div>
        </div>
      </div>
    </div>
  )
}
