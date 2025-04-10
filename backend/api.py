# Import the ESG document processor
from esg_document_processor import ESGDocumentProcessor
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import uuid
import tempfile
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="ESG Analysis API",
    description="API for ESG document analysis and insights",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ESG document processor
processor = None

# Store processing tasks and results
processing_tasks = {}
processing_results = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the ESG document processor on startup"""
    global processor
    
    # Get OpenAI API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Initialize processor
    processor = ESGDocumentProcessor(
        openai_api_key=openai_api_key,
        model="gpt-4o",
        vector_db_dir="./chroma_db",
        embedding_model="text-embedding-3-small"
    )

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"status": "ok", "message": "ESG Analysis API is running"}

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_document: bool = Form(True)
):
    """Upload and process an ESG document"""
    if not processor:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    # Check file extension
    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    
    allowed_extensions = ['.pdf', '.docx', '.doc', '.csv', '.xlsx', '.xls', '.txt']
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    try:
        # Write uploaded file to temporary file
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add task to background tasks
        background_tasks.add_task(
            process_document_task,
            task_id=task_id,
            file_path=temp_file.name,
            original_filename=filename,
            index_document=index_document
        )
        
        # Return task information
        return {
            "task_id": task_id,
            "status": "processing",
            "message": f"Document '{filename}' is being processed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

async def process_document_task(task_id: str, file_path: str, original_filename: str, index_document: bool):
    """Background task to process a document"""
    global processing_tasks, processing_results
    
    # Update task status
    processing_tasks[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "filename": original_filename,
        "started_at": datetime.now().isoformat()
    }
    
    try:
        # Process the document
        result = await processor.process_document(file_path, index_document=index_document)
        
        # Store the result
        processing_results[task_id] = result
        
        # Update task status
        processing_tasks[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "filename": original_filename,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        # Update task status on error
        processing_tasks[task_id] = {
            "status": "failed",
            "progress": 0.0,
            "filename": original_filename,
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)

@app.get("/documents/status/{task_id}")
async def get_document_status(task_id: str):
    """Get the status of a document processing task"""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found")
    
    task_info = processing_tasks[task_id]
    result = None
    error = None
    
    if task_info["status"] == "completed" and task_id in processing_results:
        result = processing_results[task_id]
    elif task_info["status"] == "failed" and "error" in task_info:
        error = task_info["error"]
    
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "result": result,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/documents/list")
async def list_documents():
    """List all processed documents"""
    if not processor:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    try:
        # Get collection stats
        stats = processor.vector_db.get_collection_stats()
        
        # Get all documents from the collection
        documents = []
        
        # For each completed task, get the document information
        for task_id, result in processing_results.items():
            if "document_id" in result and result["document_id"]:
                documents.append({
                    "document_id": result["document_id"],
                    "filename": result["file_processed"],
                    "processed_at": result["timestamp"],
                    "file_type": os.path.splitext(result["file_processed"])[1].lower(),
                    "status": "processed"
                })
        
        return documents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/documents/query")
async def query_documents(query_request: dict):
    """Query the vector database for relevant document chunks"""
    if not processor:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    try:
        # Extract parameters
        query = query_request.get("query", "")
        n_results = query_request.get("n_results", 5)
        document_id = query_request.get("document_id", None)
        
        # Check if document_id is valid, if provided
        if document_id:
            # This is a placeholder - in a real implementation, you would check
            # if the document exists in the database
            pass
        
        # Query the vector database
        results = await processor.query_documents(
            query=query,
            n_results=n_results,
            document_id=document_id
        )
        
        # Add document_id to response if it was provided
        if document_id:
            results["document_id"] = document_id
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

# Add more endpoints as needed for your application