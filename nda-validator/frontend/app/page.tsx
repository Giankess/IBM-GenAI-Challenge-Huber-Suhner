"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Upload, FileText, Check, MessageSquare, RefreshCw, Database } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { toast } from "@/components/ui/use-toast"

// API base URL - in a real app, this would come from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export default function Home() {
  const [step, setStep] = useState<"upload" | "review" | "feedback" | "complete">("upload")
  const [file, setFile] = useState<File | null>(null)
  const [feedback, setFeedback] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [documentId, setDocumentId] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`)
      }

      const data = await response.json()
      setDocumentId(data.document_id)
      setStep("review")
    } catch (error) {
      console.error("Error uploading file:", error)
      toast({
        title: "Error",
        description: "Failed to upload and analyze document. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleAccept = async () => {
    if (!documentId) return

    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/accept/${documentId}`, {
        method: "POST",
      })

      if (!response.ok) {
        throw new Error(`Failed to accept suggestions: ${response.status}`)
      }

      await response.json()
      setStep("complete")
    } catch (error) {
      console.error("Error accepting suggestions:", error)
      toast({
        title: "Error",
        description: "Failed to accept suggestions. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleFeedbackSubmit = async () => {
    if (!feedback.trim() || !documentId) return

    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          document_id: documentId,
          feedback: feedback,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to submit feedback: ${response.status}`)
      }

      await response.json()
      setStep("review")
      setFeedback("")
    } catch (error) {
      console.error("Error submitting feedback:", error)
      toast({
        title: "Error",
        description: "Failed to submit feedback. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleStartOver = () => {
    setStep("upload")
    setFile(null)
    setFeedback("")
    setDocumentId(null)
  }

  const downloadDocument = (type: "redline" | "clean") => {
    if (!documentId) return

    // Create a download link
    const downloadUrl = `${API_BASE_URL}/download/${documentId}/${type}`

    // Create a temporary anchor element and trigger the download
    const a = document.createElement("a")
    a.href = downloadUrl
    a.download = type === "redline" ? "nda_redline.docx" : "nda_clean.docx"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-8 bg-gray-50">
      <div className="w-full max-w-5xl">
        {/* Header with HUBER+SUHNER logo */}
        <div className="flex justify-between items-center mb-8 border-b pb-4">
          <div className="flex items-center">
            <Image
              src="/placeholder.svg?height=40&width=200"
              alt="HUBER+SUHNER Logo"
              width={200}
              height={40}
              className="mr-4"
            />
            <h1 className="text-2xl font-bold text-[#003366]">NDA Validator Assistant</h1>
          </div>
          <Link href="/training">
            <Button variant="outline">
              <Database className="h-4 w-4 mr-2" />
              Training Dashboard
            </Button>
          </Link>
        </div>

        <Card className="w-full shadow-lg">
          <CardHeader className="bg-[#003366] text-white">
            <CardTitle>NDA Validation Tool</CardTitle>
            <CardDescription className="text-gray-200">
              Upload your NDA document for AI-powered review and suggestions
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6">
            {step === "upload" && (
              <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-gray-300 rounded-lg">
                <Upload className="h-12 w-12 text-gray-400 mb-4" />
                <p className="mb-4 text-gray-600">Upload your NDA document (Word format)</p>
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  accept=".docx,.doc"
                  onChange={handleFileChange}
                />
                <label htmlFor="file-upload">
                  <Button asChild className="bg-[#003366] hover:bg-[#002244]">
                    <span>Select Document</span>
                  </Button>
                </label>
                {file && (
                  <div className="mt-4 flex items-center">
                    <FileText className="h-5 w-5 mr-2 text-[#003366]" />
                    <span>{file.name}</span>
                  </div>
                )}
              </div>
            )}

            {step === "review" && (
              <div className="space-y-6">
                <div className="p-4 bg-gray-100 rounded-lg">
                  <h3 className="font-medium mb-2">Document Analysis Complete</h3>
                  <p className="text-sm text-gray-600 mb-4">
                    Our AI has analyzed your NDA and created a redline version with suggested changes.
                  </p>
                  <Button variant="outline" className="mr-2" onClick={() => downloadDocument("redline")}>
                    <FileText className="h-4 w-4 mr-2" />
                    Download Redline Document
                  </Button>
                </div>

                <div className="p-4 border rounded-lg">
                  <h3 className="font-medium mb-2">What would you like to do next?</h3>
                  <p className="text-sm text-gray-600 mb-4">
                    You can accept all suggestions or provide feedback for further refinement.
                  </p>
                </div>
              </div>
            )}

            {step === "feedback" && (
              <div className="space-y-6">
                <h3 className="font-medium">Provide Feedback</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Please share your thoughts on the suggested changes. Our AI will adjust the document accordingly.
                </p>
                <Textarea
                  placeholder="Enter your feedback here..."
                  className="min-h-[150px]"
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                />
              </div>
            )}

            {step === "complete" && (
              <div className="space-y-6 text-center">
                <div className="flex justify-center">
                  <div className="rounded-full bg-green-100 p-3">
                    <Check className="h-8 w-8 text-green-600" />
                  </div>
                </div>
                <h3 className="font-medium text-lg">Process Complete</h3>
                <p className="text-gray-600">Your clean NDA document with accepted changes is ready for download.</p>
                <Button className="bg-[#003366] hover:bg-[#002244]" onClick={() => downloadDocument("clean")}>
                  <FileText className="h-4 w-4 mr-2" />
                  Download Clean Document
                </Button>
              </div>
            )}
          </CardContent>
          <CardFooter className="flex justify-between border-t pt-4">
            {step === "upload" && (
              <div className="w-full flex justify-end">
                <Button
                  className="bg-[#003366] hover:bg-[#002244]"
                  disabled={!file || isLoading}
                  onClick={handleUpload}
                >
                  {isLoading ? "Processing..." : "Upload & Analyze"}
                </Button>
              </div>
            )}

            {step === "review" && (
              <div className="w-full flex justify-between">
                <Button variant="outline" onClick={handleStartOver}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Start Over
                </Button>
                <div>
                  <Button variant="outline" className="mr-2" onClick={() => setStep("feedback")}>
                    <MessageSquare className="h-4 w-4 mr-2" />
                    Give Feedback
                  </Button>
                  <Button className="bg-[#003366] hover:bg-[#002244]" onClick={handleAccept} disabled={isLoading}>
                    {isLoading ? "Processing..." : "Accept Suggestions"}
                  </Button>
                </div>
              </div>
            )}

            {step === "feedback" && (
              <div className="w-full flex justify-between">
                <Button variant="outline" onClick={() => setStep("review")}>
                  Back
                </Button>
                <Button
                  className="bg-[#003366] hover:bg-[#002244]"
                  onClick={handleFeedbackSubmit}
                  disabled={!feedback.trim() || isLoading}
                >
                  {isLoading ? "Processing..." : "Submit Feedback"}
                </Button>
              </div>
            )}

            {step === "complete" && (
              <div className="w-full flex justify-center">
                <Button variant="outline" onClick={handleStartOver}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Start New Analysis
                </Button>
              </div>
            )}
          </CardFooter>
        </Card>
      </div>
    </main>
  )
}
