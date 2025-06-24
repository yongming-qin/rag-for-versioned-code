"""
RAG Model Evaluation Framework - Baseline vs RAG-Enhanced Comparison
====================================================================

This script implements a comprehensive evaluation framework for comparing the performance
of Large Language Models (LLMs) in two scenarios:
1. Baseline: Standard code generation without additional context
2. RAG-Enhanced: Code generation with retrieved API documentation

The framework evaluates models on Rust programming tasks and measures:
- Code solution quality
- Test program generation
- Execution success rates
- Performance differences between baseline and RAG approaches

Key Features:
- Parallel processing for efficiency
- Checkpoint-based execution with resume capability
- Comprehensive result tracking
- Environment-based configuration management

Usage:
    python eval_RAG_models.py

Author: RustEvo² Research Team
Date: 2024
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
from unit import call_LLM, run_rust_code, get_test_program_prompt, get_code_solution_prompt
from rag_unit import get_RAG_document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# API Configuration - Multiple provider support
# API_KEY = os.environ["AGICTO_API_KEY"]      # Agicto API configuration
# BASE_URL = os.environ["AGICTO_BASE_URL"]

API_KEY = os.environ["XIAOAI_API_KEY"]        # XiaoAI API configuration
BASE_URL = os.environ["XIAOAI_BASE_URL"]

# Concurrency settings
MAX_WORKERS = 5  # Maximum number of concurrent workers for parallel processing

# List of models to evaluate
MODELS = [
    "gpt-4o-mini"  # OpenAI's GPT-4 Omni Mini model for evaluation
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def needs_testing(entry: Dict[str, Any], model: str, suffix: str) -> bool:
    """
    Check if the model with given suffix (baseline/rag) needs testing.
    
    This function determines whether a specific model evaluation needs to be performed
    by checking if the required result fields are already present in the entry.
    
    Args:
        entry (Dict[str, Any]): The data entry containing evaluation results
        model (str): The model name being evaluated
        suffix (str): Either 'baseline' or 'rag' to indicate evaluation type
        
    Returns:
        bool: True if testing is needed, False if results already exist
    """
    code_solution_key = f"{model}_{suffix}_code_solution"
    test_program_key = f"{model}_{suffix}_test_program"
    result_key = f"{model}_{suffix}_result"
    
    # Return True if any of the required fields are missing
    return not entry.get(code_solution_key) or not entry.get(test_program_key) or not entry.get(result_key)

def initialize_model_fields(entry: Dict[str, Any]) -> None:
    """
    Initialize all model-specific fields in the entry dictionary.
    
    This function ensures that all expected fields for each model and evaluation type
    are present in the entry, preventing KeyError exceptions during processing.
    
    Args:
        entry (Dict[str, Any]): The data entry to initialize
    """
    for model in MODELS:
        for suffix in ['baseline', 'rag']:
            for field in ['code_solution', 'test_program', 'result']:
                key = f"{model}_{suffix}_{field}"
                entry.setdefault(key, '')

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def process_single_model(entry: Dict[str, Any], model: str, api_key: str, base_url: str, use_rag: bool) -> None:
    """
    Process a single model evaluation for either baseline or RAG-enhanced approach.
    
    This function handles the complete evaluation pipeline for a single model:
    1. Generate code solution using appropriate prompt
    2. Generate test program for the solution
    3. Execute the test program and capture results
    
    Args:
        entry (Dict[str, Any]): The data entry to process
        model (str): The model name to evaluate
        api_key (str): API key for the LLM service
        base_url (str): Base URL for the LLM service
        use_rag (bool): Whether to use RAG-enhanced approach
    """
    suffix = "rag" if use_rag else "baseline"
    query = entry.get('query', '')
    signature = entry.get('signature', '')
    api_info = entry.get('api_info', '')
    api = entry.get('api', '')

    # Enhance API info with RAG-retrieved documentation if using RAG
    if use_rag := (suffix == 'rag'):
        retrieved_docs = get_RAG_document(query)
        api_info += f"\n\nRetrieved Documentation:\n{retrieved_docs}"

    # Generate code solution using the enhanced prompt
    code_solution = call_LLM(
        get_code_solution_prompt(query, signature, api_info),
        model, api_key=API_KEY, BASE_URL=base_url
    )
    entry[f"{model}_{suffix}_code_solution"] = code_solution

    # Generate test program for the code solution
    test_program = call_LLM(
        get_test_program_prompt(query, code_solution, api),
        model, API_KEY, BASE_URL
    )
    entry[f"{model}_{suffix}_test_program"] = test_program

    # Execute the test program and capture results
    try:
        entry[f"{model}_{suffix}_result"] = run_rust_code(test_program)
    except Exception as e:
        entry[f"{model}_{suffix}_result"] = f"Execution error: {str(e)}"

def process_entry(entry: Dict[str, Any], api_key: str, base_url: str) -> Dict[str, Any]:
    """
    Process a complete entry for all models and evaluation types.
    
    This function orchestrates the evaluation of a single data entry across all
    configured models, running both baseline and RAG-enhanced evaluations.
    
    Args:
        entry (Dict[str, Any]): The data entry to process
        api_key (str): API key for the LLM service
        base_url (str): Base URL for the LLM service
        
    Returns:
        Dict[str, Any]: The processed entry with all evaluation results
    """
    # Initialize all model fields to prevent KeyError
    initialize_model_fields(entry)

    # Process each model
    for model in MODELS:
        # First run baseline evaluation
        if needs_testing(entry, model, 'baseline'):
            process_model(entry, model, api_key=api_key, base_url=base_url, suffix='baseline')

        # Then run RAG-enhanced evaluation
        if needs_testing(entry, model, 'rag'):
            process_model(entry, model, api_key=api_key, base_url=base_url, suffix='rag')

    return entry

def process_model(entry, model, api_key, base_url, suffix):
    """
    Process a single model evaluation with the specified suffix (baseline/rag).
    
    This function handles the core evaluation logic for a single model and evaluation type,
    including prompt generation, code generation, test generation, and execution.
    
    Args:
        entry: The data entry to process
        model (str): The model name to evaluate
        api_key (str): API key for the LLM service
        base_url (str): Base URL for the LLM service
        suffix (str): Either 'baseline' or 'rag' to indicate evaluation type
    """
    # Extract task information
    query = entry.get('query', '')
    signature = entry.get('signature', '')
    api_info = entry.get('api_info', '')
    api = entry.get('api', '')

    # Enhance API info with RAG documentation if using RAG approach
    if suffix == 'rag':
        retrieved_docs = get_RAG_document(query)
        prompt_api_info = f"{api_info}\n\nRetrieved Documentation:\n{retrieved_docs}"
    else:
        prompt_api_info = api_info

    # Generate code solution
    code_solution = call_LLM(
        get_code_solution_prompt(query, signature, prompt_api_info),
        model, api_key, base_url
    )
    entry[f"{model}_{suffix}_code_solution"] = code_solution

    # Generate test program
    test_program = call_LLM(
        get_test_program_prompt(query, code_solution, api),
        model, api_key, base_url
    )
    entry[f"{model}_{suffix}_test_program"] = test_program

    # Execute test program and capture results
    try:
        entry[f"{model}_{suffix}_result"] = run_rust_code(test_program)
    except Exception as e:
        entry[f"{model}_{suffix}_result"] = f"Failed to run: {str(e)}"

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function for the RAG model evaluation framework.
    
    This function orchestrates the complete evaluation process:
    1. Loads the evaluation dataset
    2. Processes entries in batches for efficiency
    3. Saves results incrementally to prevent data loss
    4. Provides progress tracking and error handling
    
    The function uses parallel processing to improve efficiency while maintaining
    data integrity through checkpoint-based saving.
    """
    # Load the evaluation dataset
    with open("example.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    
    # Process entries in batches with progress tracking
    with tqdm(total=total_entries, desc="Overall Progress") as overall_pbar:
        for round_start in range(0, total_entries, MAX_WORKERS):  # Process entries in batches
            batch_entries = data[round_start: round_start+MAX_WORKERS]

            # Concurrently handle multiple entries using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all entries in the current batch for processing
                futures = {
                    executor.submit(process_entry, entry, API_KEY, BASE_URL): entry 
                    for entry in data[round_start:round_start+5]
                }

                # Process completed futures and update progress
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        future.result()  # Wait for the entry to be processed
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                    finally:
                        overall_pbar.update(1)  # Update progress bar

            # Save data explicitly after each concurrent round to prevent data loss
            with open("example.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main() 