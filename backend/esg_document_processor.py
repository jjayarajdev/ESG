import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import tempfile
import time
import uuid
import re
import shutil
import stat

# Document processing libraries
import pdfplumber
import docx
import csv

# OpenAI integration
import openai
from openai import AsyncOpenAI
import tiktoken

# Vector database
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ESGDocumentProcessor')

class VectorDBManager:
    """
    Manages vector database operations using ChromaDB
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "esg_documents",
                 embedding_model: str = "text-embedding-3-small",
                 reset_db: bool = False):
        """
        Initialize the Vector Database Manager
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            reset_db: Whether to reset the database if it exists
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Reset database if requested
        if reset_db and os.path.exists(persist_directory):
            logger.info(f"Resetting ChromaDB at {persist_directory}")
            self._safely_remove_directory(persist_directory)
        
        # Ensure directory exists with proper permissions
        self._ensure_directory(persist_directory)
        
        # Initialize ChromaDB client with retry logic
        self.client = self._initialize_chroma_client(persist_directory)
        
        # Set up OpenAI embedding function
        self.embedding_function = self._initialize_embedding_function(embedding_model)
        
        # Get or create collection
        self.collection = self._get_or_create_collection(collection_name)
    
    def _safely_remove_directory(self, directory: str) -> None:
        """
        Safely remove a directory and its contents
        
        Args:
            directory: Directory to remove
        """
        try:
            # Make all files writable before removal
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, stat.S_IWRITE)
                    except Exception as e:
                        logger.warning(f"Could not change permissions for {file_path}: {str(e)}")
            
            # Remove the directory
            shutil.rmtree(directory)
            logger.info(f"Successfully removed directory: {directory}")
        except Exception as e:
            logger.error(f"Error removing directory {directory}: {str(e)}")
            # Try alternative removal method if shutil.rmtree fails
            try:
                import subprocess
                if os.name == 'nt':  # Windows
                    subprocess.run(['rd', '/s', '/q', directory], check=True)
                else:  # Unix/Linux
                    subprocess.run(['rm', '-rf', directory], check=True)
                logger.info(f"Successfully removed directory using subprocess: {directory}")
            except Exception as sub_e:
                logger.error(f"Failed to remove directory using subprocess: {str(sub_e)}")
    
    def _ensure_directory(self, directory: str) -> None:
        """
        Ensure a directory exists with proper permissions
        
        Args:
            directory: Directory to ensure
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Check if directory is writable
            test_file_path = os.path.join(directory, '.write_test')
            try:
                with open(test_file_path, 'w') as f:
                    f.write('test')
                os.remove(test_file_path)
            except Exception as e:
                logger.error(f"Directory {directory} is not writable: {str(e)}")
                raise PermissionError(f"Directory {directory} is not writable: {str(e)}")
            
            logger.info(f"Directory {directory} exists and is writable")
        except Exception as e:
            logger.error(f"Error ensuring directory {directory}: {str(e)}")
            raise
    
    def _initialize_chroma_client(self, persist_directory: str, max_retries: int = 3) -> chromadb.PersistentClient:
        """
        Initialize ChromaDB client with retry logic
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            max_retries: Maximum number of retries
            
        Returns:
            ChromaDB client
        """
        for attempt in range(max_retries):
            try:
                # Create settings with increased timeout
                settings = Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
                
                # Initialize client
                client = chromadb.PersistentClient(path=persist_directory, settings=settings)
                
                # Test client connection
                client.heartbeat()
                logger.info(f"Successfully connected to ChromaDB at {persist_directory}")
                return client
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} to initialize ChromaDB failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize ChromaDB after {max_retries} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached due to the raise in the loop
        raise RuntimeError("Failed to initialize ChromaDB client")
    
    def _initialize_embedding_function(self, embedding_model: str) -> Any:
        """
        Initialize OpenAI embedding function
        
        Args:
            embedding_model: OpenAI embedding model to use
            
        Returns:
            OpenAI embedding function
        """
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            # Create a custom embedding function to handle errors better
            class CustomOpenAIEmbeddingFunction:
                def __init__(self, api_key, model_name):
                    self.api_key = api_key
                    self.model_name = model_name
                    self.client = openai.OpenAI(api_key=api_key)
                    self.max_retries = 3
                    self.retry_delay = 2
                    self.max_batch_size = 16  # Limit batch size to avoid API limits

                def __call__(self, input):
                    if not input:
                        return []
                    
                    # Process in smaller batches to avoid API limits
                    all_embeddings = []
                    for i in range(0, len(input), self.max_batch_size):
                        batch = input[i:i + self.max_batch_size]
                        batch_embeddings = self._get_embeddings_with_retry(batch)
                        all_embeddings.extend(batch_embeddings)
                    
                    return all_embeddings

                def _get_embeddings_with_retry(self, texts):
                    for attempt in range(self.max_retries):
                        try:
                            # Check for empty or very large texts
                            processed_texts = []
                            for text in texts:
                                if not text or not text.strip():
                                    processed_texts.append("Empty document")
                                elif len(text) > 25000:  # Truncate very long texts
                                    processed_texts.append(text[:25000])
                                else:
                                    processed_texts.append(text)
                            
                            response = self.client.embeddings.create(
                                model=self.model_name,
                                input=processed_texts
                            )
                            
                            embeddings = [item.embedding for item in response.data]
                            return embeddings
                            
                        except Exception as e:
                            logger.warning(f"Embedding attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                            if attempt == self.max_retries - 1:
                                logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                                return [[0.0] * 1536 for _ in texts]
                            time.sleep(self.retry_delay * (2 ** attempt))
                    
                    return [[0.0] * 1536 for _ in texts]

            
            # Create custom embedding function
            embedding_function = CustomOpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=embedding_model
            )
            
            logger.info(f"Successfully initialized custom embedding function with model {embedding_model}")
            return embedding_function
            
        except Exception as e:
            logger.error(f"Error initializing embedding function: {str(e)}")
            raise
    
    def _get_or_create_collection(self, collection_name: str, max_retries: int = 3) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection with retry logic
        
        Args:
            collection_name: Name of the collection
            max_retries: Maximum number of retries
            
        Returns:
            ChromaDB collection
        """
        for attempt in range(max_retries):
            try:
                # List existing collections
                existing_collections = self.client.list_collections()
                collection_exists = any(col.name == collection_name for col in existing_collections)
                
                if collection_exists:
                    # Get existing collection
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Connected to existing collection '{collection_name}'")
                else:
                    # Create new collection
                    collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"description": "ESG document collection"}
                    )
                    logger.info(f"Created new collection '{collection_name}'")
                
                return collection
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} to get/create collection failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get/create collection after {max_retries} attempts")
                    raise
                
                # If collection might be corrupted, try to delete and recreate it
                if attempt > 0:
                    try:
                        self.client.delete_collection(collection_name)
                        logger.info(f"Deleted potentially corrupted collection '{collection_name}'")
                    except Exception as del_e:
                        logger.warning(f"Could not delete collection: {str(del_e)}")
                
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # This should never be reached due to the raise in the loop
        raise RuntimeError(f"Failed to get or create collection {collection_name}")
    
    def add_documents(self, 
                      texts: List[str], 
                      metadatas: List[Dict[str, Any]], 
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector database
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dictionaries for each chunk
            ids: Optional list of IDs for each chunk (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
            
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add documents in smaller batches to avoid API limits
        batch_size = 16  # Reduced batch size to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Process texts to handle empty or very large texts
            processed_texts = []
            for text in batch_texts:
                if not text or not text.strip():
                    processed_texts.append("Empty document")
                elif len(text) > 25000:  # Truncate very long texts
                    processed_texts.append(text[:25000])
                else:
                    processed_texts.append(text)
            
            try:
                self.collection.add(
                    documents=processed_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"Added batch of {len(batch_texts)} documents to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {str(e)}")
                # Continue with next batch instead of failing completely
                logger.warning(f"Continuing with next batch...")
        
        return ids
    
    def search(self, 
               query: str, 
               n_results: int = 5, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database
        
        Args:
            query: Query text
            n_results: Number of results to return
            filter_criteria: Optional filter criteria for the search
            
        Returns:
            Search results
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_criteria
            )
            return results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result["documents"]:
                return {
                    "document": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving document from ChromaDB: {str(e)}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector database
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting document from ChromaDB: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"document_count": 0, "collection_name": self.collection_name}


class ESGDocumentProcessor:
    """
    A class to process ESG documents (PDF, Word, CSV), extract text,
    analyze with OpenAI LLM, and compute ESG metrics.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None, 
                 model: str = "gpt-4o",
                 vector_db_dir: str = "./chroma_db",
                 embedding_model: str = "text-embedding-3-small",
                 reset_db: bool = False):
        """
        Initialize the ESG Document Processor.
        
        Args:
            openai_api_key: OpenAI API key (defaults to environment variable)
            model: OpenAI model to use for analysis
            vector_db_dir: Directory to persist vector database
            embedding_model: OpenAI embedding model to use
            reset_db: Whether to reset the vector database
        """
        # Set up OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable.")
        
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.sync_client = openai.OpenAI(api_key=self.openai_api_key)
        self.model = model
        
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception as e:
            logger.warning(f"Could not get encoding for model {model}: {str(e)}. Using cl100k_base instead.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize vector database
        try:
            self.vector_db = VectorDBManager(
                persist_directory=vector_db_dir,
                collection_name="esg_documents",
                embedding_model=embedding_model,
                reset_db=reset_db
            )
            logger.info("Successfully initialized vector database")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            raise
        
        # Define ESG metrics categories
        self.environmental_metrics = [
            "carbon_emissions", "water_usage", "waste_management", 
            "renewable_energy", "biodiversity_impact"
        ]
        
        self.social_metrics = [
            "diversity_inclusion", "employee_training", "community_investment",
            "supply_chain_audits", "human_rights"
        ]
        
        self.governance_metrics = [
            "board_independence", "executive_compensation", "transparency",
            "ethics_policies", "shareholder_rights"
        ]
        
        self.all_metrics = self.environmental_metrics + self.social_metrics + self.governance_metrics
        
        # Processing settings
        self.max_tokens_per_chunk = 1000  # For chunking large documents
        self.chunk_overlap = 200  # Overlap between chunks
        self.max_retries = 3  # For API call retries
        self.retry_delay = 2  # Seconds between retries
    
    async def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from various file types (PDF, DOCX, CSV)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # PDF processing
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
                
            # Word document processing
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
                
            # CSV processing
            elif file_extension == '.csv':
                return self._extract_from_csv(file_path)
                
            # Excel processing
            elif file_extension in ['.xlsx', '.xls']:
                return self._extract_from_excel(file_path)
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}")
            raise

    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error in DOCX extraction: {str(e)}")
            raise
    
    def _extract_from_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                
                # Convert CSV to a readable text format
                header = rows[0] if rows else []
                text = "CSV Data:\n"
                text += ", ".join(header) + "\n"
                
                for row in rows[1:]:
                    text += ", ".join(row) + "\n"
                    
                return text
        except Exception as e:
            logger.error(f"Error in CSV extraction: {str(e)}")
            raise
    
    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error in Excel extraction: {str(e)}")
            raise
    
    def _chunk_text_for_embedding(self, text: str, doc_metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks suitable for embedding with metadata
        
        Args:
            text: The text to chunk
            doc_metadata: Document metadata
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Get token count for this paragraph
            paragraph_tokens = len(self.encoding.encode(paragraph))
            
            # If adding this paragraph would exceed the chunk size, start a new chunk
            if current_length + paragraph_tokens > self.max_tokens_per_chunk and current_chunk:
                # Create chunk with metadata
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_index": chunk_index,
                    "token_count": current_length
                }
                chunks.append((chunk_text, chunk_metadata))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(self.encoding.encode(p)) for p in current_chunk)
                chunk_index += 1
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_length += paragraph_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_metadata = {
                **doc_metadata,
                "chunk_index": chunk_index,
                "token_count": current_length
            }
            chunks.append((chunk_text, chunk_metadata))
        
        return chunks
    
    async def index_document(self, file_path: str, doc_id: Optional[str] = None) -> str:
        """
        Extract text from a document and index it in the vector database
        
        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        try:
            # Extract text from document
            text = await self.extract_text_from_file(file_path)
            
            # Create document metadata
            doc_metadata = {
                "document_id": doc_id,
                "filename": os.path.basename(file_path),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "indexed_at": datetime.now().isoformat(),
                "file_path": file_path
            }
            
            # Chunk text for embedding
            chunks = self._chunk_text_for_embedding(text, doc_metadata)
            
            # Add chunks to vector database
            chunk_texts = [chunk[0] for chunk in chunks]
            chunk_metadatas = [chunk[1] for chunk in chunks]
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            self.vector_db.add_documents(
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            raise
    
    async def retrieve_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Retrieve relevant context from the vector database for a query
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Relevant context as a string
        """
        try:
            # Search for relevant chunks
            results = self.vector_db.search(query, n_results=n_results)
            
            if not results["documents"] or not results["documents"][0]:
                return ""
            
            # Format results into context
            context_parts = []
            
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                if metadata:
                    source = metadata.get("filename", "Unknown document")
                    context_parts.append(f"--- Document: {source} ---\n{doc}\n")
                else:
                    context_parts.append(f"--- Result {i+1} ---\n{doc}\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
    
    async def analyze_text_with_llm(self, text: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze text using OpenAI LLM to extract ESG metrics
        
        Args:
            text: Text content to analyze
            query: Optional query to retrieve additional context
            
        Returns:
            Dictionary of ESG metrics and scores
        """
        # Retrieve relevant context if query is provided
        additional_context = ""
        if query:
            additional_context = await self.retrieve_relevant_context(query)
            logger.info(f"Retrieved {len(additional_context.split()) if additional_context else 0} words of additional context")
        
        # Chunk the text if it's too large
        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_tokens_per_chunk * 4:  # If text is very large
            logger.info(f"Text is very large ({len(tokens)} tokens), using chunking and vector search")
            return await self._analyze_large_text(text, query)
        
        # Create system prompt for ESG analysis
        system_prompt = """
        You are an expert ESG (Environmental, Social, Governance) analyst. 
        Analyze the provided document text and extract quantitative metrics for ESG performance.
        
        For each metric, provide:
        1. A score from 0.0 to 1.0 (where 1.0 is excellent)
        2. A brief justification for the score
        3. Confidence level (high, medium, low)
        
        If a metric cannot be evaluated from the text, indicate with "N/A" and explain why.
        
        Respond in JSON format with the following structure:
        {
            "environmental": {
                "carbon_emissions": {"score": 0.7, "justification": "...", "confidence": "high"},
                "water_usage": {"score": 0.8, "justification": "...", "confidence": "medium"},
                ...
            },
            "social": {
                "diversity_inclusion": {"score": 0.6, "justification": "...", "confidence": "high"},
                ...
            },
            "governance": {
                "board_independence": {"score": 0.9, "justification": "...", "confidence": "medium"},
                ...
            }
        }
        """
        
        user_prompt = f"""
        Analyze the following document text for ESG metrics:
        
        {text}
        """
        
        # Add additional context if available
        if additional_context:
            user_prompt += f"""
            
            Additionally, consider this relevant context from similar documents:
            
            {additional_context}
            """
        
        user_prompt += f"""
        Focus on extracting quantitative metrics for:
        
        Environmental: carbon_emissions, water_usage, waste_management, renewable_energy, biodiversity_impact
        Social: diversity_inclusion, employee_training, community_investment, supply_chain_audits, human_rights
        Governance: board_independence, executive_compensation, transparency, ethics_policies, shareholder_rights
        """
        
        # Make API call with retries
        all_metrics = {}
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent results
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                try:
                    metrics = json.loads(response.choices[0].message.content)
                    all_metrics = metrics
                    break  # Success, exit retry loop
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response from OpenAI")
                    if attempt == self.max_retries - 1:
                        raise
                    
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Calculate aggregate scores
        return self._calculate_aggregate_scores(all_metrics)
    
    async def _analyze_large_text(self, text: str, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze large text by chunking and using vector search
        
        Args:
            text: Large text content to analyze
            query: Optional query to focus the analysis
            
        Returns:
            Dictionary of ESG metrics and scores
        """
        # Create temporary document ID for this analysis
        temp_doc_id = f"temp_{uuid.uuid4()}"
        
        try:
            # Create document metadata
            doc_metadata = {
                "document_id": temp_doc_id,
                "temp_analysis": True,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Chunk text for embedding
            chunks = self._chunk_text_for_embedding(text, doc_metadata)
            
            # Add chunks to vector database
            chunk_texts = [chunk[0] for chunk in chunks]
            chunk_metadatas = [chunk[1] for chunk in chunks]
            chunk_ids = [f"{temp_doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            self.vector_db.add_documents(
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Indexed large text with {len(chunks)} chunks for analysis")
            
            # Create queries for each ESG category
            category_queries = {
                "environmental": "environmental metrics carbon emissions water usage waste management renewable energy",
                "social": "social metrics diversity inclusion employee training community investment supply chain human rights",
                "governance": "governance metrics board independence executive compensation transparency ethics policies shareholder rights"
            }
            
            # If a specific query is provided, use it to focus the analysis
            if query:
                for category in category_queries:
                    category_queries[category] = f"{category_queries[category]} {query}"
            
            # Analyze each category separately
            all_metrics = {
                "environmental": {},
                "social": {},
                "governance": {}
            }
            
            for category, category_query in category_queries.items():
                # Retrieve relevant chunks for this category
                context = await self.retrieve_relevant_context(category_query, n_results=10)
                
                if not context:
                    logger.warning(f"No relevant context found for {category} category")
                    continue
                
                # Analyze this category
                category_metrics = await self._analyze_category(category, context)
                
                if category in category_metrics:
                    all_metrics[category] = category_metrics[category]
            
            # Calculate aggregate scores
            result = self._calculate_aggregate_scores(all_metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing large text: {str(e)}")
            raise
            
        finally:
            # Clean up temporary chunks
            try:
                for chunk_id in chunk_ids:
                    self.vector_db.delete_document(chunk_id)
                logger.info(f"Cleaned up temporary chunks for analysis")
            except Exception as e:
                logger.error(f"Error cleaning up temporary chunks: {str(e)}")
    
    async def _analyze_category(self, category: str, context: str) -> Dict[str, Any]:
        """
        Analyze a specific ESG category using the provided context
        
        Args:
            category: ESG category (environmental, social, governance)
            context: Relevant context for this category
            
        Returns:
            Dictionary of metrics for this category
        """
        # Create system prompt for category analysis
        system_prompt = f"""
        You are an expert ESG (Environmental, Social, Governance) analyst specializing in {category} metrics.
        Analyze the provided document text and extract quantitative metrics for {category} performance.
        
        For each metric, provide:
        1. A score from 0.0 to 1.0 (where 1.0 is excellent)
        2. A brief justification for the score
        3. Confidence level (high, medium, low)
        
        If a metric cannot be evaluated from the text, indicate with "N/A" and explain why.
        
        Respond in JSON format with the following structure:
        {{
            "{category}": {{
                "metric_name": {{"score": 0.7, "justification": "...", "confidence": "high"}},
                ...
            }}
        }}
        """
        
        # Define metrics for this category
        metrics = {
            "environmental": self.environmental_metrics,
            "social": self.social_metrics,
            "governance": self.governance_metrics
        }
        
        user_prompt = f"""
        Analyze the following document text for {category} metrics:
        
        {context}
        
        Focus on extracting quantitative metrics for: {', '.join(metrics[category])}
        """
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                try:
                    category_metrics = json.loads(response.choices[0].message.content)
                    return category_metrics
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response for {category} category")
                    if attempt == self.max_retries - 1:
                        raise
                    
            except Exception as e:
                logger.error(f"API call failed for {category} category (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))
        
        return {category: {}}
    
    def _calculate_aggregate_scores(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate aggregate scores for each ESG category
        
        Args:
            metrics: Dictionary of ESG metrics
            
        Returns:
            Dictionary with added aggregate scores
        """
        result = {"scores": {}, "metrics": metrics}
        
        # Calculate category scores
        for category in ["environmental", "social", "governance"]:
            if category in metrics:
                valid_scores = [
                    data["score"] for metric, data in metrics[category].items() 
                    if isinstance(data, dict) and "score" in data and isinstance(data["score"], (int, float))
                ]
                
                if valid_scores:
                    result["scores"][category] = sum(valid_scores) / len(valid_scores)
                else:
                    result["scores"][category] = None
        
        # Calculate overall score
        valid_category_scores = [
            score for category, score in result["scores"].items() 
            if score is not None
        ]
        
        if valid_category_scores:
            result["scores"]["overall"] = sum(valid_category_scores) / len(valid_category_scores)
        else:
            result["scores"]["overall"] = None
            
        return result
    
    async def generate_insights(self, metrics: Dict[str, Any], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate insights and recommendations based on ESG metrics
        
        Args:
            metrics: Dictionary of ESG metrics and scores
            query: Optional query to retrieve additional context
            
        Returns:
            List of insights and recommendations
        """
        # Retrieve relevant context if query is provided
        additional_context = ""
        if query:
            additional_context = await self.retrieve_relevant_context(query)
        
        system_prompt = """
        You are an expert ESG (Environmental, Social, Governance) consultant.
        Based on the provided ESG metrics and scores, generate actionable insights and recommendations.
        
        For each insight:
        1. Identify strengths and weaknesses
        2. Provide specific, actionable recommendations
        3. Suggest industry benchmarks or best practices
        
        Respond in JSON format with an array of insights:
        [
            {
                "category": "environmental|social|governance|overall",
                "type": "strength|weakness|opportunity|risk",
                "title": "Brief title",
                "description": "Detailed description",
                "recommendation": "Specific recommendation",
                "priority": "high|medium|low"
            },
            ...
        ]
        """
        
        user_prompt = f"""
        Generate insights and recommendations based on these ESG metrics:
        
        {json.dumps(metrics, indent=2)}
        """
        
        # Add additional context if available
        if additional_context:
            user_prompt += f"""
            
            Additionally, consider this relevant context from similar documents:
            
            {additional_context}
            """
        
        user_prompt += """
        
        Provide at least 1-2 insights for each category (environmental, social, governance) 
        and 1-2 overall insights. Focus on the most significant findings.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            insights = json.loads(response.choices[0].message.content)
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    async def process_document(self, file_path: str, index_document: bool = True) -> Dict[str, Any]:
        """
        Process a document file and generate ESG analysis
        
        Args:
            file_path: Path to the document file
            index_document: Whether to index the document in the vector database
            
        Returns:
            Dictionary with ESG analysis results
        """
        start_time = time.time()
        logger.info(f"Starting processing of {file_path}")
        
        try:
            # Extract text from document
            text = await self.extract_text_from_file(file_path)
            logger.info(f"Extracted {len(text)} characters from document")
            
            # Index document if requested
            doc_id = None
            if index_document:
                doc_id = await self.index_document(file_path)
                logger.info(f"Indexed document with ID: {doc_id}")
            
            # Create query from filename for context retrieval
            filename = os.path.basename(file_path)
            query = f"ESG metrics in {filename}"
            
            # Analyze text with LLM
            metrics = await self.analyze_text_with_llm(text, query=query)
            logger.info("Completed LLM analysis of document")
            
            # Generate insights
            insights = await self.generate_insights(metrics, query=query)
            logger.info(f"Generated {len(insights)} insights")
            
            # Prepare result
            result = {
                "timestamp": datetime.now().isoformat(),
                "file_processed": os.path.basename(file_path),
                "document_id": doc_id,
                "processing_time": time.time() - start_time,
                "metrics": metrics["metrics"],
                "scores": metrics["scores"],
                "insights": insights,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "file_processed": os.path.basename(file_path),
                "processing_time": time.time() - start_time,
                "status": "error",
                "error_message": str(e)
            }
    
    async def process_multiple_documents(self, file_paths: List[str], index_documents: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel
        
        Args:
            file_paths: List of paths to document files
            index_documents: Whether to index the documents in the vector database
            
        Returns:
            List of results for each document
        """
        tasks = [self.process_document(file_path, index_document=index_documents) for file_path in file_paths]
        return await asyncio.gather(*tasks)
    
    async def query_documents(self, query: str, n_results: int = 5, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the vector database for relevant document chunks
        
        Args:
            query: Query string
            n_results: Number of results to return
            document_id: Optional document ID to filter results by
        
        Returns:
            Dict with query results
        """
        if not query:
            return {
                "query": "",
                "timestamp": self._get_timestamp(),
                "results": []
            }
        
        try:
            # Prepare filter criteria if document_id is provided
            filter_criteria = None
            if document_id:
                filter_criteria = {"document_id": document_id}
            
            # Search the vector database
            results = self.vector_db.search(
                query=query,
                n_results=n_results,
                filter_criteria=filter_criteria
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results.get("documents", [[]])[0],
                    results.get("metadatas", [[]])[0],
                    results.get("distances", [[]])[0]
                )
            ):
                relevance_score = 1.0 - min(1.0, max(0.0, float(distance)))
                
                formatted_results.append({
                    "document": doc,
                    "metadata": metadata,
                    "relevance_score": relevance_score
                })
            
            # Return query results
            return {
                "query": query,
                "timestamp": self._get_timestamp(),
                "results": formatted_results
            }
        
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            # Return empty results on error
            return {
                "query": query,
                "timestamp": self._get_timestamp(),
                "results": [],
                "error": str(e)
            }
    
    def export_results(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], output_file: str) -> None:
        """
        Export results to JSON file
        
        Args:
            results: Analysis results (single document or list of documents)
            output_file: Path to output JSON file
        """
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results exported to {output_file}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format
        
        Returns:
            Current timestamp string
        """
        return datetime.now().isoformat()


# Example usage
async def main():
    # Initialize processor
    processor = ESGDocumentProcessor(reset_db=True)
    
    # Example: Create a sample document
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
        temp.write(b"""
        ESG Report 2023
        
        Environmental Performance:
        - Carbon emissions reduced by 15% year-over-year
        - Water usage efficiency improved by 8%
        - Waste recycling rate at 76%
        - Renewable energy now accounts for 35% of total energy consumption
        
        Social Initiatives:
        - Diversity in leadership positions: 38% women
        - Average employee training: 22 hours per employee annually
        - Community investment: 1.2% of profits
        - Completed supply chain audits for 75% of tier 1 suppliers
        
        Governance:
        - Board independence at 80%
        - 30% of executive compensation linked to ESG metrics
        - Bi-annual ESG reporting implemented
        - Ethics policies updated and training completed for 95% of employees
        """)
        sample_file = temp.name
    
    # Process the sample file
    result = await processor.process_document(sample_file)
    
    # Print results
    print("\nESG Scores:")
    for category, score in result["scores"].items():
        if score is not None:
            print(f"  {category.capitalize()}: {score:.2f}")
    
    print("\nInsights:")
    for insight in result["insights"]:
        print(f"  [{insight['category'].capitalize()} - {insight['type']}] {insight['title']}")
        print(f"    {insight['description']}")
        print(f"    Recommendation: {insight['recommendation']}")
        print()
    
    # Query the vector database
    query_result = await processor.query_documents("carbon emissions reduction")
    print(f"\nQuery Results for 'carbon emissions reduction':")
    for i, result in enumerate(query_result["results"]):
        print(f"  Result {i+1} (Score: {result['relevance_score']:.2f}):")
        print(f"    Source: {result['metadata'].get('filename', 'Unknown')}")
        print(f"    Excerpt: {result['document'][:150]}...")
        print()
    
    # Export results
    processor.export_results(result, "esg_analysis_results.json")
    
    # Clean up sample file
    os.remove(sample_file)

if __name__ == "__main__":
    asyncio.run(main())