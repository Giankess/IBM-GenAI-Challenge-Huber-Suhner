# NDA Validator AI Assistant

An AI-powered tool for validating, analyzing, and suggesting improvements to Non-Disclosure Agreements (NDAs). The system uses machine learning to identify problematic clauses in legal documents and suggest appropriate replacements based on training data.

![NDA Validator Screenshot](https://placeholder.svg?height=400&width=800)

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Docker Setup (Optional)](#docker-setup-optional)
- [Usage Guide](#usage-guide)
  - [NDA Validation](#nda-validation)
  - [Training Custom Models](#training-custom-models)
  - [Using Redline Documents for Training](#using-redline-documents-for-training)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
  - [NDA Analysis Process](#nda-analysis-process)
  - [Training Process](#training-process)
  - [Redline Document Parsing](#redline-document-parsing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **AI-powered NDA Analysis**: Automatically identify problematic clauses in NDAs
- **Redline Generation**: Create redline documents with suggested changes
- **Feedback Loop**: Incorporate user feedback to improve suggestions
- **Training Dashboard**: Train custom models on your own legal documents
- **Redline Document Learning**: Extract training data from existing redlined documents
- **Model Management**: Manage and deploy different model versions
- **Self-Reflection**: Validate suggestions with a second model check
- **Long-term Memory**: Store and learn from previous analyses and feedback

## Technology Stack

### Frontend
- **Next.js 14+**: React framework for building the user interface
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality UI components
- **React Hooks**: For state management

### Backend
- **Python 3.10+**: Programming language
- **FastAPI**: Modern, high-performance web framework
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: NLP models and utilities
- **python-docx**: Library for working with Word documents
- **scikit-learn**: Machine learning utilities

### AI Models
- **nlpaueb/legal-bert-base-uncased**: Pre-trained language model for legal text
- **Custom fine-tuned models**: Trained on your specific NDA corpus

## Setup Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Docker (optional)

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Giankess/IBM-GenAI-Challenge-Huber-Suhner.git
   cd nda-validator

2. Set up the Python environment:
```shellscript
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


3. Run the backend server:

```shellscript
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000)




### Frontend Setup

1. Navigate to the frontend directory:

```shellscript
cd frontend
```


2. Install dependencies:

```shellscript
npm install
```


3. Create a `.env.local` file with the following content:

```plaintext
NEXT_PUBLIC_API_URL=http://localhost:8000
```


4. Run the development server:

```shellscript
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)




### Docker Setup (Optional)

You can also run the entire application using Docker Compose:

1. Build and start the containers:

```shellscript
docker-compose up -d
```


2. The application will be available at:

1. Frontend: [http://localhost:3000](http://localhost:3000)
2. Backend API: [http://localhost:8000](http://localhost:8000)
3. API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)





## Usage Guide

### NDA Validation

1. Navigate to the main page
2. Upload an NDA document (Word format)
3. Review the AI-generated suggestions in the redline document
4. Either accept all suggestions or provide feedback
5. Download the final document


### Training Custom Models

1. Navigate to the Training Dashboard
2. Create a new dataset by uploading:

1. Word documents with [PROBLEMATIC] tags
2. JSON annotation files
3. Redline documents with tracked changes



3. Start a training job on your dataset
4. Monitor training progress
5. Activate your trained model when complete


### Using Redline Documents for Training

The system can extract training data from redline documents with tracked changes:

1. Upload redline documents with the "These are redline documents" option checked
2. The system will automatically extract:

1. Deletions (strikethroughs) as problematic clauses
2. Insertions as replacement text



3. This data is used to train the model to recognize similar patterns


## API Documentation

The backend API provides the following endpoints:

### Document Processing

- `POST /upload` - Upload and analyze an NDA document
- `GET /download/{document_id}/{type}` - Download a document (redline or clean)
- `POST /feedback` - Submit feedback on suggestions
- `POST /accept/{document_id}` - Accept all suggestions


### Training

- `POST /datasets` - Create a new training dataset
- `GET /datasets` - List all datasets
- `POST /training` - Start a training job
- `GET /training` - List all training jobs
- `GET /models` - List all model versions
- `POST /models/activate` - Activate a model version
- `POST /parse-redline` - Parse a redline document


For detailed API documentation, visit [http://localhost:8000/docs](http://localhost:8000/docs) when the backend is running.

## Project Structure

```
nda-validator/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── train.py             # Training functionality
│   ├── redline_parser.py    # Redline document parser
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend Docker configuration
├── frontend/
│   ├── app/                 # Next.js pages and components
│   ├── components/          # Reusable UI components
│   ├── public/              # Static assets
│   └── package.json         # Frontend dependencies
└── docker-compose.yml       # Docker Compose configuration
```

## How It Works

### NDA Analysis Process

1. **Document Parsing**: The system extracts text from Word documents
2. **Clause Identification**: The AI model identifies potentially problematic clauses
3. **Suggestion Generation**: The system generates suggestions for improvements
4. **Validation**: Suggestions are validated by a second model (self-reflection)
5. **Redline Creation**: A redline document is created with the suggestions
6. **Feedback Processing**: User feedback is incorporated to improve future suggestions


### Training Process

1. **Data Collection**: Legal documents are collected and labeled
2. **Feature Extraction**: Text features are extracted from the documents
3. **Model Training**: The legal-bert model is fine-tuned on the labeled data
4. **Evaluation**: The model is evaluated on a validation set
5. **Deployment**: The trained model is deployed for use in the NDA validator


## License



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
