"""
RAG (Retrieval-Augmented Generation) Unit for Rust API Documentation - Updated Version
====================================================================================

This module implements a comprehensive RAG system for retrieving relevant API documentation
to enhance LLM performance on Rust programming tasks. This version uses the latest
LangChain functions and patterns.

Key Features:
- Multi-provider LLM and embedding model support using latest LangChain patterns
- Vector database management with Chroma
- Document processing and chunking
- Similarity search for relevant documentation
- Configurable model selection for different providers

Supported Providers:
- OpenAI (GPT models, text-embedding models)
- Anthropic (Claude models)
- Google (Gemini, Vertex AI)
- Azure (OpenAI services)
- AWS (Bedrock)
- Groq (Llama models)
- NVIDIA (AI models)
- Together AI (Open source models)
- HuggingFace (Local models)
- Ollama (Local models)

Usage:
    from RAG_unit_updated import get_RAG_document
    relevant_docs = get_RAG_document("how to use HashMap in Rust")

Author: RustEvo² Research Team
Date: 2024
Updated: 2024 (Latest LangChain patterns)
"""

import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata

# Suppress deprecation warnings for a cleaner output
# warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# LLM MODEL SELECTION FUNCTIONS - UPDATED FOR LATEST LANGCHAIN
# =============================================================================

def select_chat_model(model="OpenAI", version=""):
    """
    Select and initialize a chat model from various providers using latest LangChain patterns.
    
    This function supports multiple LLM providers and handles API key management
    for each provider. It uses the latest LangChain import patterns and initialization
    methods.
    
    Args:
        model (str): The model provider name (e.g., "OpenAI", "Anthropic", "Groq")
        version (str): Optional version specification (currently unused)
        
    Returns:
        The initialized chat model instance
        
    Raises:
        Exception: If the model provider is not supported or API key is missing
    """
    import getpass
    import os

    # Groq provider configuration - Updated pattern
    if model == "Groq":
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

        from langchain_groq import ChatGroq
        llm = ChatGroq(
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama3-8b-8192"
        )

    # OpenAI provider configuration - Updated pattern
    if model == "OpenAI":
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = ""

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini"
        )

    # Anthropic provider configuration - Updated pattern
    if model == 'Anthropic':
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-3-5-sonnet-latest"
        )

    # Azure OpenAI provider configuration - Updated pattern
    if model == 'Azure':
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )

    # Google Vertex AI provider configuration - Updated pattern
    if model == "Google Vertex":
        # Ensure your VertexAI credentials are configured
        from langchain_google_vertexai import ChatVertexAI
        llm = ChatVertexAI(
            model="gemini-2.0-flash-001",
            project="your-project-id",  # Replace with actual project ID
            location="us-central1"      # Replace with actual location
        )

    # AWS Bedrock provider configuration - Updated pattern
    if model == "AWS":
        # Ensure your AWS credentials are configured
        from langchain_aws import BedrockChat
        llm = BedrockChat(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-east-1"  # Replace with your AWS region
        )

    # NVIDIA provider configuration - Updated pattern
    if model == "NVIDIA":
        if not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key=os.environ["NVIDIA_API_KEY"]
        )

    # Together AI provider configuration - Updated pattern
    if model == "Together AI":
        if not os.environ.get("TOGETHER_API_KEY"):
            os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

        from langchain_together import Together
        llm = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            together_api_key=os.environ["TOGETHER_API_KEY"]
        )

    return llm

# =============================================================================
# EMBEDDING MODEL SELECTION FUNCTIONS - UPDATED FOR LATEST LANGCHAIN
# =============================================================================

def select_embeddings_model(model="OpenAI"):
    """
    Select and initialize an embedding model from various providers using latest LangChain patterns.
    
    This function supports multiple embedding model providers and handles
    API key management for each provider. It uses the latest LangChain import patterns.
    
    Args:
        model (str): The embedding model provider name (e.g., "OpenAI", "Azure", "Google")
        
    Returns:
        The initialized embedding model instance
        
    Raises:
        Exception: If the model provider is not supported or API key is missing
    """
    import getpass
    import os

    # OpenAI embedding model configuration - Updated pattern
    if model == "OpenAI":
        os.environ["OPENAI_API_KEY"] = ""

        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

        # Alternative configuration with custom API base
        # embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     openai_api_base="https://api.agicto.cn/v1",
        #     openai_api_key=os.environ["OPENAI_API_KEY"]
        # )

    # Azure OpenAI embedding model configuration - Updated pattern
    if model == "Azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )

    # Google Vertex AI embedding model configuration - Updated pattern
    if model == "Google":
        from langchain_google_vertexai import VertexAIEmbeddings
        embeddings = VertexAIEmbeddings(
            model="text-embedding-004",
            project="your-project-id",  # Replace with actual project ID
            location="us-central1"      # Replace with actual location
        )

    # AWS Bedrock embedding model configuration - Updated pattern
    if model == "AWS":
        from langchain_aws import BedrockEmbeddings
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"  # Replace with your AWS region
        )

    # HuggingFace embedding model configuration (local) - Updated pattern
    if model == "HuggingFace":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # or 'cuda' for GPU
        )

    # Ollama embedding model configuration (local) - Updated pattern
    if model == "Ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model="llama3",
            base_url="http://localhost:11434"  # Default Ollama URL
        )

    return embeddings

# =============================================================================
# CORE RAG DOCUMENT RETRIEVAL FUNCTION - UPDATED FOR LATEST LANGCHAIN
# =============================================================================

def get_RAG_document(query: str = "", embedding_model="OpenAI", chat_model='OpenAI') -> str:
    """
    Retrieve relevant API documentation using RAG (Retrieval-Augmented Generation).
    
    This function implements the core RAG pipeline using the latest LangChain patterns:
    1. Load and process API documentation from JSON files
    2. Split documents into manageable chunks
    3. Create or load vector embeddings in Chroma database
    4. Perform similarity search to find relevant documentation
    5. Generate a contextual response using the retrieved documents
    
    Args:
        query (str): The search query to find relevant API documentation
        embedding_model (str): The embedding model provider to use
        chat_model (str): The chat model provider to use for response generation
        
    Returns:
        str: Generated response based on retrieved relevant documentation
        
    Note:
        The function uses a persistent Chroma database to avoid re-embedding
        documents on subsequent calls, improving performance significantly.
    """
    from pprint import pprint
    
    # Import required LangChain components using latest patterns
    from langchain_community.document_loaders import JSONLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate  # Updated import

    # Define the path to the API documentation JSON file
    file_path = "./data/final_docs.json"

    def metadata_func(record: dict, metadata: dict) -> dict:
        """
        Extract and process metadata from API documentation records.
        
        This function processes both standard library APIs and third-party crate APIs,
        extracting relevant metadata fields and converting complex types to strings
        for proper storage in the vector database.
        
        Args:
            record (dict): The API documentation record
            metadata (dict): The metadata dictionary to populate
            
        Returns:
            dict: Processed metadata dictionary
        """
        # Handle standard library APIs (no crate field)
        if "crate" not in record:
            metadata["name"] = record.get("name")
            metadata["from_version"] = record.get("from_version")
            metadata["to_version"] = record.get("to_version")
            metadata["module"] = record.get("module")
            metadata["type"] = record.get("type")
            metadata["signature"] = record.get("signature")
            metadata["documentation"] = record.get("documentation")
            metadata["source_code"] = record.get("source_code")
            
            # Convert complex types to strings for storage
            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
                elif value is None:
                    metadata[key] = ""
        else:
            # Handle third-party crate APIs
            metadata["crate"] = record.get("crate")
            metadata["name"] = record.get("name")
            metadata["from_version"] = record.get("from_version")
            metadata["to_version"] = record.get("to_version")
            metadata["module"] = record.get("module")
            metadata["type"] = record.get("type")
            metadata["signature"] = record.get("signature")
            metadata["documentation"] = record.get("documentation")
            metadata["source_code"] = record.get("source_code")
            
            # Convert complex types to strings for storage
            for key, value in list(metadata.items()):
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
                elif value is None:
                    metadata[key] = ""
        
        return metadata

    # Load API documentation from JSON file
    loader = JSONLoader(
        file_path="./data/final_docs.json",
        jq_schema=".[]",  # Iterate over each object in the array
        metadata_func=metadata_func,
        text_content=False  # Preserve JSON object as string in page_content
    )

    docs = loader.load()

    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum chunk size in characters
        chunk_overlap=200,    # Overlap between chunks to maintain context
        add_start_index=True  # Add start index for better tracking
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Optional: Filter complex metadata (commented out)
    # filtered_splits = [filter_complex_metadata(doc) for doc in all_splits]
    
    # Initialize embedding model
    embeddings = select_embeddings_model(model=embedding_model)

    # Initialize or load Chroma vector database
    vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    # Check if the vector store already has documents
    existing_ids = vector_store.get()["ids"]  # Get the list of document IDs in the store

    if not existing_ids:
        # If no documents exist, embed and save them
        ids = vector_store.add_documents(documents=all_splits)
        print(f"Added documents with IDs: {ids}")
    else:
        # If documents exist, just load the existing store
        print(f"Loaded existing store with document IDs")

    # Define the prompt template for generating responses - Updated pattern
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context: {context}\nQuestion: {question}")

    # Perform similarity search to find relevant documents
    question = query
    retrieved_docs = vector_store.similarity_search(question)
    
    # Combine retrieved document content
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Create the prompt with context and question - Updated pattern
    prompt_value = prompt.invoke({"question": question, "context": docs_content})
    prompt_str = prompt_value.to_string()

    # Initialize chat model and generate response
    llm = select_chat_model(model=chat_model)

    try:
        # Updated invocation pattern for latest LangChain
        answer = llm.invoke(prompt_str)
        # Extract content from the response
        if hasattr(answer, 'content'):
            answer = answer.content
        elif hasattr(answer, 'text'):
            answer = answer.text
    except AttributeError:
        # Fallback for older LangChain versions or misconfigured LLMs
        answer = llm(prompt_str)
    
    return answer

# =============================================================================
# TESTING AND UTILITY FUNCTIONS
# =============================================================================

def test_get_RAG_document():
    """
    Test function for the RAG document retrieval system.
    
    This function provides a simple way to test the RAG system by
    querying for specific API documentation and printing the results.
    
    Usage:
        test_get_RAG_document()
    """
    context = get_RAG_document("what is hash_one API", "OpenAI", "OpenAI")
    print(context)

# =============================================================================
# MODULE DOCUMENTATION
# =============================================================================

"""
Example Usage:
-------------

# Basic usage with default models
relevant_docs = get_RAG_document("how to use HashMap in Rust")

# Custom model selection
relevant_docs = get_RAG_document(
    query="how to use HashMap in Rust",
    embedding_model="OpenAI",
    chat_model="Anthropic"
)

# Test the system
test_get_RAG_document()

Configuration:
--------------
The module uses environment variables for API keys:
- OPENAI_API_KEY: For OpenAI models
- ANTHROPIC_API_KEY: For Anthropic models
- GROQ_API_KEY: For Groq models
- AZURE_OPENAI_API_KEY: For Azure OpenAI
- NVIDIA_API_KEY: For NVIDIA models
- TOGETHER_API_KEY: For Together AI models

Data Requirements:
-----------------
The system expects a JSON file at "./data/final_docs.json" containing
API documentation records with the following structure:
{
    "name": "API name",
    "from_version": "1.71.0",
    "to_version": "1.84.0",
    "module": "std::collections",
    "type": "function",
    "signature": "fn new() -> Self",
    "documentation": "API documentation...",
    "source_code": "fn new() -> Self { ... }"
}

Performance Notes:
-----------------
- The vector database is persisted to "./chroma_db" for reuse
- Document embedding is only performed once per document
- Subsequent queries use the cached embeddings for faster retrieval
- Chunk size and overlap can be adjusted for different use cases

LangChain Version Compatibility:
-------------------------------
This updated version uses the latest LangChain patterns:
- langchain_core.prompts for prompt templates
- Provider-specific imports (langchain_openai, langchain_anthropic, etc.)
- Updated model initialization patterns
- Latest response handling methods
""" 