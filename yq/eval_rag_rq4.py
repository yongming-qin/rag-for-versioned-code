"""
RAG-Enhanced Evaluation Framework for Research Question 4 (RQ4)
================================================================

This script implements a Retrieval-Augmented Generation (RAG) based evaluation system
for assessing LLM performance on Rust API evolution tasks. It compares baseline LLM
performance with RAG-enhanced performance by retrieving relevant API documentation
and incorporating it into the generation prompts.

Key Features:
- Parallel processing of multiple LLM models
- RAG document retrieval for enhanced context
- Comprehensive evaluation metrics (Pass@1, API usage accuracy, coverage)
- Checkpoint-based processing with resume capability
- Support for multiple Rust versions

Usage:
    python eval_RAG_rq4.py --file_a ./data/final_dataset.json --file_b ./data/final_dataset.json 
    --output ./data/RQ4/rq4_results.json --api_key YOUR_API_KEY --base_url YOUR_BASE_URL

Author: RustEvo² Research Team
Date: 2024
"""

import json
import os
import subprocess
import tempfile
import re
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from rag_unit import RagDocument
from collections import defaultdict


#yq Initialize the RagDocument
rag_document = RagDocument()

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# List of LLM models to evaluate in the RAG-enhanced framework
MODELS = [
    "gpt-4.1-nano",
    # "gpt-4o",                    # OpenAI's GPT-4 Omni model
    # "o1-mini",                 # Anthropic's o1-mini model (commented out)
    # "gemini-1.5-pro",           # Google's Gemini 1.5 Pro model
    # "claude-3-5-sonnet-20240620", # Anthropic's Claude 3.5 Sonnet
    # "qwen2.5-72b-instruct",     # Alibaba's Qwen 2.5 72B model
    # "Llama-3.1-70b",            # Meta's Llama 3.1 70B model
    # "deepseek-v3",              # DeepSeek's v3 model
    # "grok-3",                   # xAI's Grok-3 model
    # "claude-3-7-sonnet-20250219" # Anthropic's Claude 3.7 Sonnet
]

# API configuration - can be set via environment variables or command line arguments
# API_KEY = os.getenv('API_KEY', '')
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL')

# =============================================================================
# CORE LLM INTERACTION FUNCTIONS
# =============================================================================

def call_llm(prompt: str, model: str, api_key: str, base_url: str) -> str:
    """
    Call the LLM API and return the response.
    
    Args:
        prompt (str): The input prompt to send to the LLM
        model (str): The model name to use for generation
        api_key (str): API key for authentication
        base_url (str): Base URL for the API endpoint
        
    Returns:
        str: The generated response from the LLM
        
    Raises:
        Exception: If the API call fails
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Moderate creativity for code generation
        )        
        code = response.choices[0].message.content.strip()
        return code
        
    except Exception as e:
        print(f"Error calling LLM {model}: {str(e)}")
        return ""

# =============================================================================
# CODE EXTRACTION AND VALIDATION FUNCTIONS
# =============================================================================

def extract_rust_code(response: str) -> str:
    """
    Extract Rust code from the LLM response.
    
    This function handles multiple formats of code responses:
    1. Code blocks marked with ```rust or ```
    2. Function definitions without code blocks
    3. Fallback to raw response if no structured code is found
    
    Args:
        response (str): The raw response from the LLM
        
    Returns:
        str: Extracted Rust code
    """
    # Look for code between ```rust and ``` markers
    rust_pattern = r"```(?:rust)?\s*([\s\S]*?)```"
    matches = re.findall(rust_pattern, response)
    
    if matches:
        return matches[0].strip()
    
    # If no code blocks found, try to extract any code-like content
    lines = response.strip().split('\n')
    code_lines = []
    in_fn_block = False
    
    for line in lines:
        if re.match(r'\s*fn\s+\w+', line):
            in_fn_block = True
        
        if in_fn_block:
            code_lines.append(line)
            
            if line.strip() == "}" and code_lines:
                break
    
    return '\n'.join(code_lines) if code_lines else response

def check_function_signature(code: str, signature: str) -> bool:
    """
    Check if the generated code contains a function matching the required signature.
    
    This validation ensures that the generated code implements the expected function
    interface. It performs a relaxed check focusing on function name matching,
    which is suitable for preliminary screening.
    
    Args:
        code (str): The generated Rust code
        signature (str): The expected function signature
        
    Returns:
        bool: True if the function signature is correct, False otherwise
    """
    import re
    
    # 如果签名为空或代码为空，直接返回False
    if not signature or not code:
        return False
    
    # 清理签名，移除属性、文档注释和可见性修饰符
    clean_signature = re.sub(r'#\[.*?\]', '', signature)
    clean_signature = re.sub(r'///.*?\n', '', clean_signature)
    clean_signature = re.sub(r'//.*?\n', '', clean_signature)
    clean_signature = re.sub(r'pub\s+', '', clean_signature).strip()
    
    # 提取函数名
    fn_match = re.search(r'fn\s+(\w+)', clean_signature)
    if not fn_match:
        return False
    
    fn_name = fn_match.group(1)
    
    # 基本检查：代码中必须有这个函数名
    if not re.search(r'fn\s+' + re.escape(fn_name) + r'\s*[(<]', code):
        return False
    
    # 更宽松的检查：只要函数名匹配，就认为签名正确
    # 这种方法适合初步筛选，在大多数情况下函数名匹配就足够了
    return True

def check_api_usage(code: str, api_name: str) -> bool:
    """
    Check if the generated code uses the specified API.
    
    This function validates that the generated code actually uses the target API
    that the task is designed to test. It checks for both full API paths and
    just the function/method names.
    
    Args:
        code (str): The generated Rust code
        api_name (str): The API name to check for usage
        
    Returns:
        bool: True if the API is used, False otherwise
    """
    import re
    
    # 如果API名称为空或代码为空，直接返回False
    if not api_name or not code:
        return False
    
    # Extract the function name from a full signature if needed
    fn_match = re.search(r'fn\s+(\w+)', api_name)
    base_api_name = fn_match.group(1) if fn_match else api_name.split('::')[-1]

    # Match just the function name usage
    result = re.search(r'\b' + re.escape(base_api_name) + r'\b', code) is not None
    print(f"check_api_usage result: {result=} {api_name=} {base_api_name=} {code=}")
    
    input()
    return result

    ############# original #############
    
    # 获取API的最后部分(函数/方法名)
    base_api_name = api_name.split('::')[-1] if '::' in api_name else api_name
    
    # 仅做最基本检查，并打印结果但总是返回True
    full_match = re.search(r'\b' + re.escape(api_name) + r'\b', code)
    base_match = re.search(r'\b' + re.escape(base_api_name) + r'\b', code)
    
    if full_match or base_match:
        return True
    else:
        return False

# =============================================================================
# TEST EXECUTION FUNCTIONS
# =============================================================================

def create_test_file(code: str, test_program: str) -> str:
    """
    Combine code solution and test program into a complete test file.
    
    This function creates a complete Rust test file by combining the generated
    code solution with the test program, ensuring proper module structure.
    
    Args:
        code (str): The generated Rust code solution
        test_program (str): The test program to validate the solution
        
    Returns:
        str: Complete Rust test file content
    """
    # Wrap test program in appropriate module if not already done
    if "#[cfg(test)]" not in test_program:
        test_program = f"""
#[cfg(test)]
mod tests {{
    use super::*;
    
    {test_program}
}}
"""
    
    return f"{code}\n\n{test_program}"

def run_rust_test(test_file_content: str, rust_version: str = "1.84.0") -> Tuple[bool, str]:
    """
    Run Rust test file with the specified version and return success status and error message.
    
    This function compiles and executes the test file using the specified Rust version,
    providing detailed error information if the test fails.
    
    Args:
        test_file_content (str): The complete Rust test file content
        rust_version (str): The Rust version to use for compilation
        
    Returns:
        Tuple[bool, str]: (success_status, error_message)
    """
    with tempfile.NamedTemporaryFile(suffix='.rs', delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(test_file_content.encode('utf-8'))
    
    try:
        # Remove extension and add new extension
        base_path = os.path.splitext(temp_file_path)[0]
        output_path = f"{base_path}.exe" if os.name == 'nt' else base_path
        
        # Compile command with specific Rust version
        compile_cmd = f'rustup run {rust_version} rustc --test "{temp_file_path}" -o "{output_path}"'
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            return False, f"Compilation error: {compile_result.stderr}"
        
        # Run the compiled test
        test_cmd = f'"{output_path}"'
        test_result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if test_result.returncode != 0:
            return False, f"Test execution error: {test_result.stderr}"
        
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Timeout: Test took too long to execute"
    except Exception as e:
        return False, f"Error running tests: {str(e)}"
    finally:
        # Clean up temporary files
        try:
            os.remove(temp_file_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass

# =============================================================================
# PROMPT GENERATION FUNCTIONS
# =============================================================================

def get_code_generation_prompt_rag(query: str, api_info: Dict[str, Any], function_signature: str) -> str:
    """
    Create a prompt for code generation with RAG-enhanced context.
    
    This function generates a comprehensive prompt that includes:
    1. The original task description
    2. Required function signature
    3. Retrieved documentation from RAG system
    4. Specific requirements for implementation
    
    Args:
        query (str): The original programming task description
        api_info (Dict[str, Any]): API information dictionary
        function_signature (str): The required function signature
        
    Returns:
        str: Formatted prompt for code generation
    """
    # Retrieve relevant documentation using RAG
    retrieved_docs = rag_document.get_rag_document(query)
    
    prompt = f"""
    You are an expert Rust programmer. Write a Rust function implementation for the following task:

    Task Description:
    {query}

    Required Function Signature:
    ```rust
    {function_signature}
    ```

    Retrieved Documentation:
    {retrieved_docs}

    Requirements:
    1. Implement ONLY the function with the given signature, no additional functions.
    2. Make sure your code is compatible with relevant Rust version (must later than 1.71.0)
    3. Do not include tests, main function, or any code outside the required function.
    4. Do not include additional comments or explanations.
    5. Import the necessary modules for the function implementation at the top of the file.

    Respond with ONLY the Rust function implementation, nothing else.    

"""
    return prompt

def get_code_generation_prompt_for_ground_truth(query: str, api_info: Dict[str, Any], function_signature: str) -> str:
    prompt = f"""
        You are an expert Rust programmer. Write a Rust function implementation for the following task:

        Task Description:
        {query}

        Required Function Signature:
        ```rust
        {function_signature}
        ```
        
        Retrieved Documentation:
        {api_info}

        Requirements:
        1. Implement ONLY the function with the given signature, no additional functions.
        2. Make sure your code is compatible with relevant Rust version (must later than 1.71.0)
        3. Do not include tests, main function, or any code outside the required function.
        4. Do not include additional comments or explanations.
        5. Import the necessary modules for the function implementation at the top of the file.

        Respond with ONLY the Rust function implementation, nothing else.    

    """
    return prompt

def get_code_generation_prompt_no_rag(query: str, api_info: Dict[str, Any], function_signature: str) -> str:
    prompt = f"""
        You are an expert Rust programmer. Write a Rust function implementation for the following task:

        Task Description:
        {query}

        Required Function Signature:
        ```rust
        {function_signature}
        ```

        Requirements:
        1. Implement ONLY the function with the given signature, no additional functions.
        2. Make sure your code is compatible with relevant Rust version (must later than 1.71.0)
        3. Do not include tests, main function, or any code outside the required function.
        4. Do not include additional comments or explanations.
        5. Import the necessary modules for the function implementation at the top of the file.

        Respond with ONLY the Rust function implementation, nothing else.    

    """
    return prompt

# =============================================================================
# TASK PROCESSING FUNCTIONS
# =============================================================================

def process_task(test_type: str, task_id: int, task_a: Dict[str, Any], task_b: Dict[str, Any], model: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """
    Process a single task for a specific model with RAG enhancement.
    
    This function handles the complete pipeline for a single task:
    1. Generate code using RAG-enhanced prompt
    2. Validate function signature
    3. Check API usage
    4. Execute tests
    5. Return comprehensive results
    
    Args:
        task_a (Dict[str, Any]): Task data from file A (queries, signatures, tests)
        task_b (Dict[str, Any]): Task data from file B (API information)
        model (str): The LLM model to use
        api_key (str): API key for the LLM service
        base_url (str): Base URL for the LLM service
        
    Returns:
        Dict[str, Any]: Task results with model-specific fields
    """
    result = task_a.copy()  # Start with a copy of the original task_a data
    
    # Extract required fields
    query = task_a.get("query")
    function_signature = task_a.get("function_signature")
    test_program = task_a.get("test_program")
    
    print(f"query {task_id}: {query}\n--------------------------------\n")
    print(f"ground truth function_signature: {function_signature}\n--------------------------------\n")
    
    
    # Generate code using RAG-enhanced prompt
    #yq The below does use the task_b information to generate the prompt. which makes sense because the task_b can be seen as the ground truth
    if test_type == "no_rag":
        prompt = get_code_generation_prompt_no_rag(query, task_b, function_signature)
    elif test_type == "rag":
        prompt = get_code_generation_prompt_rag(query, task_b, function_signature)
    elif test_type == "ground_truth":
        prompt = get_code_generation_prompt_for_ground_truth(query, task_b, function_signature)
    raw_response = call_llm(prompt, model, api_key, base_url)
    code = extract_rust_code(raw_response)
    print(f"llm generated code: {code}\n--------------------------------\n")
    
    # Check function signature
    if not check_function_signature(code, function_signature):
        #yq I guess this means the rag does not get the correct function signature
        result[f"RAG_{model}_code"] = "INCORRECT SIG"
        result[f"RAG_{model}_test_result"] = "FAILED"
        print(f"test result: {result[f'RAG_{model}_test_result']=}\n {result[f'RAG_{model}_code']=}")
        return result
    
    # Check API usage
    # api_name = task_b.get("name") #yq ground truth api name
    # if not check_api_usage(code, api_name):
    #     #yq I guess this means the rag does not get the correct api chunks
    #     result[f"RAG_{model}_code"] = "INCORRECT API"
    #     result[f"RAG_{model}_test_result"] = "FAILED"
    #     print(f"test result: {result[f'RAG_{model}_test_result']=}\n {result[f'RAG_{model}_code']=}")
    #     return result
    
    # Determine Rust version to use based on task information
    rust_version = "1.84.0"  # Default version
    if "crate" not in task_b:
        to_version = task_b.get("to_version")
        if to_version:
            rust_version = to_version
    
    # Run test to validate the solution
    test_file = create_test_file(code, test_program)
    success, error_message = run_rust_test(test_file, rust_version)
    print(f"test result: {success=}, compilation error: {error_message=}\n")
    
    # if error_message contains something like rustup toolchain install 1.78.0-x86_64-unknown-linux-gnu
    # run rustup toolchain install xx-unknown-linux-gnu with subprocess
    if "rustup toolchain install" in error_message:
        try:
            # Extract and sanitize the toolchain name
            match = re.search(r"rustup toolchain install ([^\s`'\"]+)", error_message)
            if match:
                rust_version = match.group(1)
                subprocess.run(["rustup", "toolchain", "install", rust_version], check=True)
                success, error_message = run_rust_test(test_file, rust_version)
                print(f"test result: {success=}, compilation error: {error_message=}\n")
            else:
                print("Could not parse toolchain version from error message.")
        except (subprocess.CalledProcessError, IndexError) as e:
            print(f"Failed to install Rust toolchain {rust_version}: {e}")
            # Continue with the original error
    
    # Set result fields
    result[f"rag_{model}_code"] = code
    result[f"rag_{model}_test_result"] = "SUCCESS" if success else "FAILED"
    result[f"rag_{model}_error_message"] = error_message
    
    # input("end of process_task")
    
    return result


def print_summary_statistics(output_file, models: List[str]):
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # Calculate and print success rates per model
    for model in models:
        success_count = sum(1 for result in results if result.get(f"rag_{model}_test_result", "") == "SUCCESS")
        success_rate = (success_count / len(results)) * 100 if results else 0
        incorrect_sig = sum(1 for result in results if result.get(f"rag_{model}_code", "") == "INCORRECT SIG")
        incorrect_api = sum(1 for result in results if result.get(f"rag_{model}_code", "") == "INCORRECT API")
        incorrect_error_message = sum(1 for result in results if result.get(f"rag_{model}_error_message", "") != "")
        
        print(f"\nModel: {model}")
        print(f"Success rate: {success_rate:.2f}% ({success_count}/{len(results)})")
        print(f"Incorrect signatures: {incorrect_sig}")
        print(f"Incorrect API usage: {incorrect_api}")
        print(f"Incorrect error message: {incorrect_error_message}")
        

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_all_models(test_type: str,
                       file_a_data: List[Dict[str, Any]], file_b_data: Dict[str, str],
                       models: List[str], api_key: str, base_url: str, output_file: str, max_workers: int = 4):
    """
    Process all tasks for all models in parallel with checkpoint support.
    
    This function implements the main evaluation pipeline with the following features:
    1. Parallel processing of multiple models
    2. Checkpoint-based processing with resume capability
    3. Progress tracking with tqdm
    4. Comprehensive result collection and statistics
    
    Args:
        file_a_data (List[Dict[str, Any]]): List of tasks from file A
        file_b_data (Dict[str, str]): API information from file B
        models (List[str]): List of models to evaluate
        api_key (str): API key for LLM service
        base_url (str): Base URL for LLM service
        output_file (str): Path to save results
        max_workers (int): Maximum number of concurrent workers
    """
    results = []
    processed_task_ids = set()
    
    # 检查输出文件是否存在，如果存在则加载已有结果
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                # 记录已经处理过的任务ID
                for result in results:
                    processed_task_ids.add(result.get("task_idx"))
                print(f"Loaded {len(results)} existing results from {output_file}")
    except Exception as e:
        print(f"Error loading existing results: {str(e)}")
        # 如果加载失败，使用空列表开始
        results = []
        processed_task_ids = set()
    
    print(f"processed_task_ids: {processed_task_ids}")
    # 创建任务ID到task_b数据的映射，以便快速查找
    #yq api documentation
    task_b_mapping = {idx + 1: item for idx, item in enumerate(file_b_data)}
    print(f"length of task_b_mapping: {len(task_b_mapping)}")
        
    # 过滤出尚未处理的任务
    remaining_tasks = [task for task in file_a_data if task.get("task_idx") not in processed_task_ids]
    remaining_tasks = remaining_tasks[:50]
    print(f"number of remaining tasks: {len(remaining_tasks)}")
    
    # Process tasks with progress tracking
    with tqdm(total=len(remaining_tasks) * len(models), desc="Processing tasks") as pbar:
        for task_a in remaining_tasks:
            task_id = task_a.get("task_idx")
            
            # 跳过没有匹配的task_b的任务
            if task_id not in task_b_mapping:
                print(f"Warning: No matching data found in file B for task_id {task_id}")
                continue
            
            task_b = task_b_mapping[task_id]
            task_result = task_a.copy()
            
            # 为每个模型并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_task, test_type, task_id, task_a, task_b, model, api_key, base_url): model
                    for model in models
                }
                
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        model_result = future.result()
                        # 更新任务结果，添加模型特定的字段
                        task_result[f"rag_{model}_code"] = model_result.get(f"rag_{model}_code")
                        task_result[f"rag_{model}_test_result"] = model_result.get(f"rag_{model}_test_result")
                        task_result[f"rag_{model}_error_message"] = model_result.get(f"rag_{model}_error_message")
                    except Exception as e:
                        print("ERROR: " + str(e))
                        task_result[f"rag_{model}_code"] = f"ERROR: {str(e)}"
                        task_result[f"rag_{model}_test_result"] = "FAILED"
                        task_result[f"rag_{model}_error_message"] = f"ERROR: {str(e)}"
                    finally:
                        pbar.update(1)
                    tmp_code, tmp_test_result, tmp_error_message = task_result[f'rag_{model}_code'], task_result[f'rag_{model}_test_result'], task_result[f'rag_{model}_error_message']
                    print(f"result {task_id=} {model=} {tmp_code=}, {tmp_test_result=}, {tmp_error_message=}")
                    print("================================================"*10)
            
            results.append(task_result)
            
            # 每处理10个任务保存一次检查点
            if len(results) % 5 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final save of all results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print(f"\nProcessing complete. Results saved to: {output_file}")
    
    print_summary_statistics(output_file, models)
    



# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function for the RAG-enhanced evaluation framework.
    
    This function handles command-line argument parsing, data loading,
    and orchestrates the complete evaluation process.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM models on Rust API evolution tasks with RAG enhancement")
    parser.add_argument("--test_type", required=True, choices=["no_rag", "rag", "ground_truth"], help="Type of test to run")
    parser.add_argument("--file_a", required=True, help="Input JSON file A with tasks and test programs")
    parser.add_argument("--file_b", required=True, help="Input JSON file B with API information")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to evaluate")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of concurrent workers")
    parser.add_argument("--api_key", help="API key for LLM service")
    parser.add_argument("--base_url", help="Base URL for LLM service")
    
    args = parser.parse_args()
    
    # Use provided API credentials if available
    api_key = args.api_key if args.api_key else API_KEY
    base_url = args.base_url if args.base_url else BASE_URL
    # print(f"api_key: {api_key}")
    # print(f"base_url: {base_url}")
    
    # Check if API key is provided
    if api_key == "your-api-key-here":
        print("Warning: No API key provided. Please set the API_KEY environment variable or use --api_key.")
    
    # Load the data from files
    try:
        with open(args.file_a, "r", encoding="utf-8") as f:
            file_a_data = json.load(f)
        
        with open(args.file_b, "r", encoding="utf-8") as f:
            file_b_data = json.load(f)
            
        print(f"Loaded {len(file_a_data)} tasks from file A with tasks and test programs and {len(file_b_data)} API entries from file B with API information")
    except Exception as e:
        print(f"Error loading input files: {str(e)}")
        sys.exit(1)
        
    # print_summary_statistics(args.output, args.models)
    
    # Process all models with the complete evaluation pipeline
    output_file = args.output.split(".")[0] + f"_{args.test_type}.json"
    print(f"output_file: {output_file}")

    process_all_models(
        args.test_type,
        file_a_data, 
        file_b_data, 
        args.models, 
        api_key, 
        base_url, 
        output_file, 
        args.max_workers
    )

if __name__ == "__main__":
    main() 