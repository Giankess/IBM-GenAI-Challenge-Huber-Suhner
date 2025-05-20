"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Upload, Play, BarChart, Save, AlertCircle } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import Image from "next/image"
import Link from "next/link"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { toast } from "@/components/ui/use-toast"

interface TrainingDataset {
  id: string
  name: string
  document_count: number
  created_at: string
  is_redline?: boolean
}

interface TrainingJob {
  id: string
  dataset_id: string
  status: "running" | "completed" | "failed" | "queued"
  created_at: string
  completed_at?: string
  metrics?: {
    accuracy: number
    precision: number
    recall: number
    f1_score: number
  }
}

interface ModelVersion {
  id: string
  name: string
  training_job_id: string
  accuracy: number
  created_at: string
  is_active: boolean
}

interface RedlineParseResult {
  total_clauses: number
  problematic_clauses: number
  data: {
    clauses: Array<{
      text: string
      is_problematic: boolean
      replacement?: string
    }>
  }
}

// API base URL - in a real app, this would come from environment variables
const API_BASE_URL = "http://localhost:8000"

export default function TrainingPage() {
  const [activeTab, setActiveTab] = useState("datasets")
  const [datasets, setDatasets] = useState<TrainingDataset[]>([])
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([])
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isLoading, setIsLoading] = useState({
    datasets: false,
    jobs: false,
    models: false,
  })
  const [datasetName, setDatasetName] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [isRedline, setIsRedline] = useState(false)
  const [redlineParseResult, setRedlineParseResult] = useState<RedlineParseResult | null>(null)
  const [showRedlinePreview, setShowRedlinePreview] = useState(false)
  const [isParsingRedline, setIsParsingRedline] = useState(false)

  // Fetch data when the component mounts and when the active tab changes
  useEffect(() => {
    if (activeTab === "datasets") {
      fetchDatasets()
    } else if (activeTab === "jobs") {
      fetchTrainingJobs()
    } else if (activeTab === "models") {
      fetchModelVersions()
    }
  }, [activeTab])

  const fetchDatasets = async () => {
    setIsLoading((prev) => ({ ...prev, datasets: true }))
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`)
      if (!response.ok) {
        throw new Error(`Failed to fetch datasets: ${response.status}`)
      }
      const data = await response.json()
      setDatasets(data)
    } catch (error) {
      console.error("Error fetching datasets:", error)
      toast({
        title: "Error",
        description: "Failed to load datasets. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading((prev) => ({ ...prev, datasets: false }))
    }
  }

  const fetchTrainingJobs = async () => {
    setIsLoading((prev) => ({ ...prev, jobs: true }))
    try {
      const response = await fetch(`${API_BASE_URL}/training`)
      if (!response.ok) {
        throw new Error(`Failed to fetch training jobs: ${response.status}`)
      }
      const data = await response.json()
      setTrainingJobs(data)
    } catch (error) {
      console.error("Error fetching training jobs:", error)
      toast({
        title: "Error",
        description: "Failed to load training jobs. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading((prev) => ({ ...prev, jobs: false }))
    }
  }

  const fetchModelVersions = async () => {
    setIsLoading((prev) => ({ ...prev, models: true }))
    try {
      const response = await fetch(`${API_BASE_URL}/models`)
      if (!response.ok) {
        throw new Error(`Failed to fetch model versions: ${response.status}`)
      }
      const data = await response.json()
      setModelVersions(data)
    } catch (error) {
      console.error("Error fetching model versions:", error)
      toast({
        title: "Error",
        description: "Failed to load model versions. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading((prev) => ({ ...prev, models: false }))
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFiles(e.target.files)

      // If redline is selected and we have a file, parse it to preview
      if (isRedline && e.target.files.length === 1) {
        parseRedlineFile(e.target.files[0])
      } else {
        setRedlineParseResult(null)
        setShowRedlinePreview(false)
      }
    }
  }

  const parseRedlineFile = async (file: File) => {
    setIsParsingRedline(true)
    setShowRedlinePreview(false)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${API_BASE_URL}/parse-redline`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Failed to parse redline document: ${response.status}`)
      }

      const result = await response.json()
      setRedlineParseResult(result)
      setShowRedlinePreview(true)
    } catch (error) {
      console.error("Error parsing redline document:", error)
      toast({
        title: "Error",
        description: "Failed to parse redline document. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsParsingRedline(false)
    }
  }

  const handleUploadDataset = async () => {
    if (!datasetName.trim() || !selectedFiles || selectedFiles.length === 0) return

    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append("name", datasetName)
      formData.append("is_redline", isRedline.toString())

      for (let i = 0; i < selectedFiles.length; i++) {
        formData.append("files", selectedFiles[i])
      }

      const response = await fetch(`${API_BASE_URL}/datasets`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Failed to upload dataset: ${response.status}`)
      }

      const newDataset = await response.json()

      toast({
        title: "Success",
        description: "Dataset uploaded successfully.",
      })

      // Refresh the datasets list
      fetchDatasets()

      // Reset form
      setDatasetName("")
      setSelectedFiles(null)
      setRedlineParseResult(null)
      setShowRedlinePreview(false)

      // Reset the file input
      const fileInput = document.getElementById("file-upload") as HTMLInputElement
      if (fileInput) fileInput.value = ""
    } catch (error) {
      console.error("Error uploading dataset:", error)
      toast({
        title: "Error",
        description: "Failed to upload dataset. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  const startTraining = async (datasetId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/training`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset_id: datasetId,
          epochs: 3,
          batch_size: 8,
          learning_rate: 2e-5,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to start training: ${response.status}`)
      }

      const result = await response.json()

      toast({
        title: "Success",
        description: "Training job started successfully.",
      })

      // Refresh the training jobs list
      fetchTrainingJobs()
    } catch (error) {
      console.error("Error starting training:", error)
      toast({
        title: "Error",
        description: "Failed to start training job. Please try again.",
        variant: "destructive",
      })
    }
  }

  const activateModel = async (modelId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/models/activate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_version_id: modelId,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to activate model: ${response.status}`)
      }

      const result = await response.json()

      toast({
        title: "Success",
        description: "Model activated successfully.",
      })

      // Refresh the model versions list
      fetchModelVersions()
    } catch (error) {
      console.error("Error activating model:", error)
      toast({
        title: "Error",
        description: "Failed to activate model. Please try again.",
        variant: "destructive",
      })
    }
  }

  // Calculate job progress based on status
  const getJobProgress = (job: TrainingJob) => {
    if (job.status === "completed") return 100
    if (job.status === "failed") return 100
    if (job.status === "queued") return 0
    // For running jobs, we don't have a progress indicator from the API
    // In a real implementation, you would get this from the API
    return 50
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

                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="is-redline"
                        checked={isRedline}
                        onCheckedChange={(checked) => {
                          setIsRedline(checked === true)
                          if (!checked) {
                            setRedlineParseResult(null)
                            setShowRedlinePreview(false)
                          }
                        }}
                      />
                      <Label htmlFor="is-redline">These are redline documents with tracked changes</Label>
                    </div>

                    {isRedline && (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Redline Document Processing</AlertTitle>
                        <AlertDescription>
                          The system will automatically extract problematic clauses (deletions) and their replacements
                          (insertions) from redline documents with tracked changes.
                        </AlertDescription>
                      </Alert>
                    )}

                    <div>
                      <Label htmlFor="file-upload">
                        {isRedline ? "Upload Redline Documents" : "Upload Labeled Documents"}
                      </Label>
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
                        {isRedline
                          ? "Upload Word documents with tracked changes (redlines)"
                          : "Upload Word documents with labeled problematic clauses or JSON annotation files"}
                      </p>
                    </div>

                    {isParsingRedline && (
                      <div className="text-center py-4">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-[#003366] mx-auto mb-2"></div>
                        <p className="text-sm text-gray-600">Parsing redline document...</p>
                      </div>
                    )}

                    {showRedlinePreview && redlineParseResult && (
                      <div className="border rounded-lg p-4 bg-gray-50">
                        <h4 className="font-medium mb-2">Redline Document Preview</h4>
                        <div className="space-y-2">
                          <p className="text-sm">
                            Found <span className="font-bold">{redlineParseResult.problematic_clauses}</span>{" "}
                            problematic clauses with replacements out of {redlineParseResult.total_clauses} total
                            clauses.
                          </p>

                          <div className="max-h-60 overflow-y-auto border rounded bg-white p-2">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Problematic Clause</TableHead>
                                  <TableHead>Replacement</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {redlineParseResult.data.clauses
                                  .filter((clause) => clause.is_problematic)
                                  .map((clause, index) => (
                                    <TableRow key={index}>
                                      <TableCell className="text-red-600 line-through">{clause.text}</TableCell>
                                      <TableCell className="text-green-600">{clause.replacement}</TableCell>
                                    </TableRow>
                                  ))}
                              </TableBody>
                            </Table>
                          </div>
                        </div>
                      </div>
                    )}

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
                  {isLoading.datasets ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#003366] mx-auto mb-4"></div>
                      <p>Loading datasets...</p>
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Documents</TableHead>
                          <TableHead>Created</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {datasets.map((dataset) => (
                          <TableRow key={dataset.id}>
                            <TableCell className="font-medium">{dataset.name}</TableCell>
                            <TableCell>
                              {dataset.is_redline ? (
                                <span className="px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                                  Redline
                                </span>
                              ) : (
                                <span className="px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
                                  Standard
                                </span>
                              )}
                            </TableCell>
                            <TableCell>{dataset.document_count}</TableCell>
                            <TableCell>{new Date(dataset.created_at).toLocaleDateString()}</TableCell>
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
                            <TableCell colSpan={5} className="text-center py-4 text-gray-500">
                              No datasets available. Upload your first dataset to get started.
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                  )}
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
                {isLoading.jobs ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#003366] mx-auto mb-4"></div>
                    <p>Loading training jobs...</p>
                  </div>
                ) : (
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
                        const dataset = datasets.find((d) => d.id === job.dataset_id)
                        return (
                          <TableRow key={job.id}>
                            <TableCell className="font-medium">{dataset?.name || job.dataset_id}</TableCell>
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
                                <Progress value={getJobProgress(job)} className="h-2" />
                                <span className="text-xs text-gray-500 mt-1">{getJobProgress(job)}%</span>
                              </div>
                            </TableCell>
                            <TableCell>{new Date(job.created_at).toLocaleDateString()}</TableCell>
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
                )}
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
                {isLoading.models ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#003366] mx-auto mb-4"></div>
                    <p>Loading model versions...</p>
                  </div>
                ) : (
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
                          <TableCell>{new Date(model.created_at).toLocaleDateString()}</TableCell>
                          <TableCell>
                            {model.is_active ? (
                              <span className="px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">Active</span>
                            ) : (
                              <span className="px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">Inactive</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {model.is_active ? (
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
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
