"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Upload, Play, BarChart, Save } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface TrainingDataset {
  id: string
  name: string
  documentCount: number
  createdAt: string
}

interface TrainingJob {
  id: string
  datasetId: string
  status: "queued" | "running" | "completed" | "failed"
  progress: number
  createdAt: string
  completedAt?: string
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1Score: number
  }
}

interface ModelVersion {
  id: string
  name: string
  trainingJobId: string
  accuracy: number
  createdAt: string
  isActive: boolean
}

export default function TrainingPage() {
  const [activeTab, setActiveTab] = useState("datasets")
  const [datasets, setDatasets] = useState<TrainingDataset[]>([])
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([])
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [datasetName, setDatasetName] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)

  // Mock data for demonstration
  useEffect(() => {
    // In a real implementation, these would be fetched from the API
    setDatasets([
      { id: "ds1", name: "Corporate NDAs", documentCount: 24, createdAt: "2025-05-10" },
      { id: "ds2", name: "Vendor Agreements", documentCount: 18, createdAt: "2025-05-12" },
    ])

    setTrainingJobs([
      {
        id: "job1",
        datasetId: "ds1",
        status: "completed",
        progress: 100,
        createdAt: "2025-05-11",
        completedAt: "2025-05-11",
        metrics: {
          accuracy: 0.89,
          precision: 0.92,
          recall: 0.87,
          f1Score: 0.89,
        },
      },
      {
        id: "job2",
        datasetId: "ds2",
        status: "running",
        progress: 65,
        createdAt: "2025-05-14",
      },
    ])

    setModelVersions([
      {
        id: "model1",
        name: "NDA-Validator-v1.2",
        trainingJobId: "job1",
        accuracy: 0.89,
        createdAt: "2025-05-11",
        isActive: true,
      },
    ])
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(e.target.files)
    }
  }

  const handleUploadDataset = async () => {
    if (!datasetName.trim() || !selectedFiles || selectedFiles.length === 0) return

    setIsUploading(true)

    // In a real implementation, you would upload the files to your Python backend
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Add the new dataset to the list
      const newDataset: TrainingDataset = {
        id: `ds${datasets.length + 1}`,
        name: datasetName,
        documentCount: selectedFiles.length,
        createdAt: new Date().toISOString().split("T")[0],
      }

      setDatasets([...datasets, newDataset])
      setDatasetName("")
      setSelectedFiles(null)

      // Reset the file input
      const fileInput = document.getElementById("file-upload") as HTMLInputElement
      if (fileInput) fileInput.value = ""
    } catch (error) {
      console.error("Error uploading dataset:", error)
    } finally {
      setIsUploading(false)
    }
  }

  const startTraining = async (datasetId: string) => {
    // In a real implementation, you would call your backend API to start training
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Add a new training job
      const newJob: TrainingJob = {
        id: `job${trainingJobs.length + 1}`,
        datasetId,
        status: "queued",
        progress: 0,
        createdAt: new Date().toISOString().split("T")[0],
      }

      setTrainingJobs([...trainingJobs, newJob])
    } catch (error) {
      console.error("Error starting training:", error)
    }
  }

  const activateModel = async (modelId: string) => {
    // In a real implementation, you would call your backend API to activate the model
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Update model versions to set the selected one as active
      const updatedModels = modelVersions.map((model) => ({
        ...model,
        isActive: model.id === modelId,
      }))

      setModelVersions(updatedModels)
    } catch (error) {
      console.error("Error activating model:", error)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-8 bg-gray-50">
      <div className="w-full max-w-6xl">
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
            <h1 className="text-2xl font-bold text-[#003366]">NDA Validator Training</h1>
          </div>
          <Link href="/">
            <Button variant="outline">Back to Validator</Button>
          </Link>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="datasets">Training Datasets</TabsTrigger>
            <TabsTrigger value="jobs">Training Jobs</TabsTrigger>
            <TabsTrigger value="models">Model Versions</TabsTrigger>
          </TabsList>

          {/* Datasets Tab */}
          <TabsContent value="datasets">
            <Card>
              <CardHeader className="bg-[#003366] text-white">
                <CardTitle>Training Datasets</CardTitle>
                <CardDescription className="text-gray-200">
                  Upload and manage labeled NDA documents for model training
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 space-y-6">
                <div className="border rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-4">Upload New Dataset</h3>
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="dataset-name">Dataset Name</Label>
                      <Input
                        id="dataset-name"
                        placeholder="e.g., Corporate NDAs 2025"
                        value={datasetName}
                        onChange={(e) => setDatasetName(e.target.value)}
                      />
                    </div>

                    <div>
                      <Label htmlFor="file-upload">Upload Labeled Documents</Label>
                      <div className="mt-1 flex items-center">
                        <input
                          type="file"
                          id="file-upload"
                          className="hidden"
                          multiple
                          accept=".docx,.doc,.json"
                          onChange={handleFileChange}
                        />
                        <label htmlFor="file-upload" className="cursor-pointer">
                          <div className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50">
                            <Upload className="h-4 w-4 mr-2" />
                            Select Files
                          </div>
                        </label>
                        <span className="ml-3 text-sm text-gray-500">
                          {selectedFiles ? `${selectedFiles.length} files selected` : "No files selected"}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-gray-500">
                        Upload Word documents with labeled problematic clauses or JSON annotation files
                      </p>
                    </div>

                    <Button
                      className="bg-[#003366] hover:bg-[#002244]"
                      disabled={!datasetName.trim() || !selectedFiles || selectedFiles.length === 0 || isUploading}
                      onClick={handleUploadDataset}
                    >
                      {isUploading ? "Uploading..." : "Upload Dataset"}
                    </Button>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-medium mb-4">Available Datasets</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Documents</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {datasets.map((dataset) => (
                        <TableRow key={dataset.id}>
                          <TableCell className="font-medium">{dataset.name}</TableCell>
                          <TableCell>{dataset.documentCount}</TableCell>
                          <TableCell>{dataset.createdAt}</TableCell>
                          <TableCell>
                            <Button variant="outline" size="sm" onClick={() => startTraining(dataset.id)}>
                              <Play className="h-4 w-4 mr-2" />
                              Train Model
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                      {datasets.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={4} className="text-center py-4 text-gray-500">
                            No datasets available. Upload your first dataset to get started.
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Training Jobs Tab */}
          <TabsContent value="jobs">
            <Card>
              <CardHeader className="bg-[#003366] text-white">
                <CardTitle>Training Jobs</CardTitle>
                <CardDescription className="text-gray-200">Monitor and manage model training jobs</CardDescription>
              </CardHeader>
              <CardContent className="pt-6">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Dataset</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead>Metrics</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trainingJobs.map((job) => {
                      const dataset = datasets.find((d) => d.id === job.datasetId)
                      return (
                        <TableRow key={job.id}>
                          <TableCell className="font-medium">{dataset?.name || job.datasetId}</TableCell>
                          <TableCell>
                            <span
                              className={`px-2 py-1 rounded-full text-xs ${
                                job.status === "completed"
                                  ? "bg-green-100 text-green-800"
                                  : job.status === "running"
                                    ? "bg-blue-100 text-blue-800"
                                    : job.status === "queued"
                                      ? "bg-yellow-100 text-yellow-800"
                                      : "bg-red-100 text-red-800"
                              }`}
                            >
                              {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                            </span>
                          </TableCell>
                          <TableCell>
                            <div className="w-full max-w-xs">
                              <Progress value={job.progress} className="h-2" />
                              <span className="text-xs text-gray-500 mt-1">{job.progress}%</span>
                            </div>
                          </TableCell>
                          <TableCell>{job.createdAt}</TableCell>
                          <TableCell>
                            {job.metrics ? (
                              <Button variant="outline" size="sm">
                                <BarChart className="h-4 w-4 mr-2" />
                                View Metrics
                              </Button>
                            ) : (
                              <span className="text-gray-500 text-sm">Not available</span>
                            )}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                    {trainingJobs.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-4 text-gray-500">
                          No training jobs found. Start a new training job from the Datasets tab.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Model Versions Tab */}
          <TabsContent value="models">
            <Card>
              <CardHeader className="bg-[#003366] text-white">
                <CardTitle>Model Versions</CardTitle>
                <CardDescription className="text-gray-200">Manage and deploy trained model versions</CardDescription>
              </CardHeader>
              <CardContent className="pt-6">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Accuracy</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {modelVersions.map((model) => (
                      <TableRow key={model.id}>
                        <TableCell className="font-medium">{model.name}</TableCell>
                        <TableCell>{(model.accuracy * 100).toFixed(1)}%</TableCell>
                        <TableCell>{model.createdAt}</TableCell>
                        <TableCell>
                          {model.isActive ? (
                            <span className="px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">Active</span>
                          ) : (
                            <span className="px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">Inactive</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {model.isActive ? (
                            <Button variant="outline" size="sm" disabled>
                              <Save className="h-4 w-4 mr-2" />
                              Current Model
                            </Button>
                          ) : (
                            <Button variant="outline" size="sm" onClick={() => activateModel(model.id)}>
                              <Save className="h-4 w-4 mr-2" />
                              Activate
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                    {modelVersions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center py-4 text-gray-500">
                          No model versions available. Complete a training job to create a model version.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
