"""
RAG (Retrieval-Augmented Generation) Unit for Rust API Documentation
====================================================================

This module implements a comprehensive RAG system for retrieving relevant API documentation
to enhance LLM performance on Rust programming tasks. The system uses vector embeddings
and similarity search to find the most relevant API documentation for a given query.

Key Features:
- Multi-provider LLM and embedding model support
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
    from RAG_unit import get_RAG_document
    relevant_docs = get_RAG_document("how to use HashMap in Rust")

Author: RustEvo² Research Team
Date: 2024
"""

import json
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores.utils import filter_complex_metadata

# Suppress deprecation warnings for a cleaner output
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# LLM MODEL SELECTION FUNCTIONS
# =============================================================================

def select_chat_model(model="OpenAI", version=""):
    """
    Select and initialize a chat model from various providers.
    
    This function supports multiple LLM providers and handles API key management
    for each provider. It initializes the appropriate model based on the provider
    and configuration.
    
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

    # Groq provider configuration
    if model == "Groq":
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("llama3-8b-8192", model_provider="groq")

    # OpenAI provider configuration
    if model == "OpenAI":
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set")

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    # Anthropic provider configuration
    if model == 'Anthropic':
        if not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")

    # Azure OpenAI provider configuration
    if model == 'Azure':
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    # Google Vertex AI provider configuration
    if model == "Google Vertex":
        # Ensure your VertexAI credentials are configured
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")

    # AWS Bedrock provider configuration
    if model == "AWS":
        # Ensure your AWS credentials are configured
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")

    # NVIDIA provider configuration
    if model == "NVIDIA":
        if not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("meta/llama3-70b-instruct", model_provider="nvidia")

    # Together AI provider configuration
    if model == "Together AI":
        if not os.environ.get("TOGETHER_API_KEY"):
            os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together AI: ")

        from langchain.chat_models import init_chat_model
        llm = init_chat_model("mistralai/Mixtral-8x7B-Instruct-v0.1", model_provider="together")

    return llm

# =============================================================================
# EMBEDDING MODEL SELECTION FUNCTIONS
# =============================================================================

def select_embeddings_model(model="OpenAI"):
    """
    Select and initialize an embedding model from various providers.
    
    This function supports multiple embedding model providers and handles
    API key management for each provider. It initializes the appropriate
    embedding model based on the provider configuration.
    
    Args:
        model (str): The embedding model provider name (e.g., "OpenAI", "Azure", "Google")
        
    Returns:
        The initialized embedding model instance
        
    Raises:
        Exception: If the model provider is not supported or API key is missing
    """
    import getpass
    import os

    # OpenAI embedding model configuration
    if model == "OpenAI":
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set")

        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Alternative configuration with custom API base (commented out)
        # from langchain_openai import OpenAIEmbeddings
        # embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-large",
        #     openai_api_base="https://api.agicto.cn/v1"
        # )

    # Azure OpenAI embedding model configuration
    if model == "Azure":
        if not os.environ.get("AZURE_OPENAI_API_KEY"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

        from langchain_openai import AzureOpenAIEmbeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    # Google Vertex AI embedding model configuration
    if model == "Google":
        from langchain_google_vertexai import VertexAIEmbeddings
        embeddings = VertexAIEmbeddings(model="text-embedding-004")

    # AWS Bedrock embedding model configuration
    if model == "AWS":
        from langchain_aws import BedrockEmbeddings
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    # HuggingFace embedding model configuration (local)
    if model == "HuggingFace":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Ollama embedding model configuration (local)
    if model == "Ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="llama3")

    return embeddings

# =============================================================================
# CORE RAG DOCUMENT RETRIEVAL FUNCTION
# =============================================================================

def get_RAG_document(query: str = "", embedding_model="OpenAI", chat_model='OpenAI') -> str:
    """
    Retrieve relevant API documentation using RAG (Retrieval-Augmented Generation).
    
    This function implements the core RAG pipeline:
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
    
    # Import required LangChain components
    from langchain_community.document_loaders import JSONLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain.prompts import ChatPromptTemplate

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
            # metadata["examples"] = record.get("examples")  # Commented out
            metadata["source_code"] = record.get("source_code")
            
            #yq no change_type and old_source_code compare to the api doc
            
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
    # file_path is the path to the API documentation JSON file
    loader = JSONLoader(
        file_path="APIDocs.json",
        jq_schema=".[]",  # Iterate over each object in the array
        metadata_func=metadata_func,
        text_content=False  # Preserve JSON object as string in page_content
    )

    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["doc_id"] = i  # or use a unique hash or UUID

    # Split documents into smaller chunks for better retrieval
    #yq In text_splitter.split_documents(docs), the page_content of each Document is what gets split.
    #yq The metadata of each Document is kept in the metadata of each split Document.
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
        # ids = vector_store.add_documents(documents=filtered_splits)  # Alternative
        print(f"Added documents with {len(ids)} IDs.")
    else:
        # If documents exist, just load the existing store
        # print(f"Loaded existing store with document IDs: {existing_ids}")
        print(f"Loaded existing store with document IDs")

    # Define the prompt template for generating responses
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context: {context}\nQuestion: {question}\n")

    # Perform similarity search to find relevant documents
    question = query
    retrieved_docs = vector_store.similarity_search(question)
    
    # Combine retrieved document content
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    if False:
        print(f"For question: {question}\n The retrieved documents are:\n")
        for doc in retrieved_docs:
            try:
                # Parse the JSON string
                parsed = json.loads(doc.page_content)
                print(f"parsed: {parsed}")
                # Extract and print the source_code with real newlines
                print("source_code:")
                print(parsed["source_code"])  # This will interpret \n as actual line breaks
                print("--------------------------------")
            except json.JSONDecodeError as e:
                print("⚠️ JSON parse error:", e)
                print("Raw doc.page_content:\n", doc.page_content)
                print("--------------------------------")

    # Create the prompt with context and question
    prompt_value = prompt.invoke({"question": question, "context": docs_content})
    prompt_str = prompt_value.to_string()

    # Initialize chat model and generate response
    llm = select_chat_model(model=chat_model)

    try:
        answer = llm.invoke(prompt_str)
    except AttributeError:
        # Fallback for older LangChain versions or misconfigured LLMs
        answer = llm(prompt_str)
    
    print("\n\nanswer:", answer.content)  # Debug output (commented out)
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
    # context = get_RAG_document("what is hash_one API", "OpenAI", "OpenAI")
    context = get_RAG_document("what is is_empty API", "OpenAI", "OpenAI")
    # print(context)

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
""" 

if __name__ == "__main__":
    test_get_RAG_document()