# Full implementation using pretrained MonoBERT for reranking and GRPO-style training

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import dotenv
from collections import defaultdict
from eval_rag_rq4 import get_code_generation_prompt_rag, get_code_generation_prompt_no_rag, call_llm, extract_rust_code, create_test_file, run_rust_test


class GrpoLearningMonoBERT():
    def __init__(self, query_items, api_items):
        # Use pretrained cross-encoder MonoBERT
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        # Data setup
        self.query_items = query_items
        self.api_items = api_items
        self.query_texts = [item["query"] for item in query_items]
        self.function_signatures = [item["function_signature"] for item in query_items]
        self.test_programs = [item["test_program"] for item in query_items]
        self.api_texts = [str(item) for item in api_items]
        self.bm25 = BM25Okapi([text.split() for text in self.api_texts])

        # LLM setup
        dotenv.load_dotenv()
        self.llm_model = "gpt-4.1-nano"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        
        # Memory setup
        if not os.path.exists("memory.json"):
            self.memory_prompts = defaultdict(lambda: defaultdict(lambda: [None, None, None])) # [prompt, llm_response, extracted_code]
        else:
            with open("memory.json", "r", encoding="utf-8") as f:
                loaded_memory = json.load(f)
                # Convert to defaultdict to prevent KeyError
                self.memory_prompts = defaultdict(dict, loaded_memory)
                for idx, value in self.memory_prompts.items():
                    for mode, prompt in value.items():
                        print(f"idx: {idx}, mode: {mode}, length of prompt: {len(prompt)}")
                
    def __del__(self):
        with open("memory.json", "w") as f:
            json.dump(self.memory_prompts, f)

    def grpo_loss(self, current_probs, old_probs, advantages, epsilon=0.2):
        ratio = current_probs / (old_probs + 1e-8)
        clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        return -torch.mean(torch.min(ratio * advantages, clipped * advantages))

    def simulate_reward(self, chunk):
        return random.choice([0.0, 0.5, 1.0])  # Replace with actual compiler feedback

    def train(self, epochs=3, top_n=10):
        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for idx, query in enumerate(self.query_texts):
                self.optimizer.zero_grad()

                scores = self.bm25.get_scores(query.split())
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
                top_docs = [self.api_texts[i] for i in top_indices]

                pairs = [f"{query} [SEP] {doc}" for doc in top_docs]
                tokenized = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**tokenized)
                logits = outputs.logits.squeeze()
                probs = torch.softmax(logits, dim=0)
                old_probs = probs.detach()

                rewards = torch.tensor([self.simulate_reward(doc) for doc in top_docs])
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                loss = self.grpo_loss(probs, old_probs, advantages)
                loss.backward()
                self.optimizer.step()

                print(f"[{idx}] Loss: {loss.item():.4f}")

    def rerank_and_test(self, item_idx, mode="bm25", use_trained_model=False, top_n=10, retrieve_k=4):
        query = self.query_texts[item_idx]
        function_signature = self.function_signatures[item_idx]
        test_program = self.test_programs[item_idx]

        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        bm25_top_docs = [self.api_texts[i] for i in top_indices]

        if mode == "bm25":
            selected_docs = bm25_top_docs[:retrieve_k]
            prompt = get_code_generation_prompt_rag(query, selected_docs, function_signature)
        elif mode == "monobert" or mode == "grpo":
            self.model.eval()
            with torch.no_grad():
                pairs = [f"{query} [SEP] {doc}" for doc in bm25_top_docs]
                tokenized = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
                outputs = self.model(**tokenized)
                logits = outputs.logits.squeeze()
                scores = logits if logits.ndim == 1 else logits[:, 0]
                ranked_indices = torch.argsort(scores, descending=True)
                reranked_docs = [bm25_top_docs[i] for i in ranked_indices]
                selected_docs = reranked_docs[:retrieve_k]

            prompt = get_code_generation_prompt_rag(query, selected_docs, function_signature)
            # print(f"\n[{mode.upper()}] Prompt:\n{prompt_rag}\n")
        elif mode == "no_rag":
            prompt = get_code_generation_prompt_no_rag(query, function_signature)
            
        raw_response = call_llm(prompt, self.llm_model, self.api_key, self.base_url)
        code = extract_rust_code(raw_response)
        # print(f"Generated Code:\n{code}\n")
        
        if str(item_idx) not in self.memory_prompts or mode not in self.memory_prompts[str(item_idx)]:
            print(f"Prompt {item_idx} {mode} not in memory_prompts")
            input("Press Enter to continue...")
            self.memory_prompts[str(item_idx)][mode] = [prompt, raw_response, code]
        else:
            if prompt != self.memory_prompts[str(item_idx)][mode][0]:
                print(f"Prompt changed for {item_idx} in {mode} mode")
                print(f"Old prompt: {self.memory_prompts[item_idx][mode][0]}")
                print(f"New prompt: {prompt}")
                input("Press Enter to continue...")
            elif code != self.memory_prompts[str(item_idx)][mode][2]:
                print(f"Code changed for {item_idx} in {mode} mode")
                print(f"Old code: {self.memory_prompts[item_idx][mode][2]}")
                print(f"New code: {code}")
                input("Press Enter to continue...")
                
            

        test_file = create_test_file(code, test_program)
        success, error_message = run_rust_test(test_file, "1.84.0")
        print(f"Test result: {success=}, Compilation Error: {error_message}\n")

        return success


def main():
    with open("RustEvo^2.json", "r", encoding="utf-8") as f:
        query_data = json.load(f)
    with open("APIDocs.json", "r", encoding="utf-8") as f:
        api_data = json.load(f)

    grpo = GrpoLearningMonoBERT(query_data, api_data)

    # Test all 3 modes on one sample
    n_success_no_rag = 0
    n_success_bm25 = 0
    n_success_monobert = 0
    success_no_rag_list = []
    bm25_monobert_different_cases = []
    success_no_rag_but_failed_in_bm25_or_monobert = []
    for idx in range(0, 5):
        print(f"idx: {idx}")
        success_no_rag = grpo.rerank_and_test(idx, mode="no_rag")
        n_success_no_rag += success_no_rag
        success_no_rag_list.append(success_no_rag)
        print("--------------------------------"*5)
        
        if False:
            success_bm25 = grpo.rerank_and_test(idx, mode="bm25")
            n_success_bm25 += success_bm25
            print("--------------------------------"*5)
        
            success_monobert = grpo.rerank_and_test(idx, mode="monobert")
            n_success_monobert += success_monobert
            print("================================================"*5)
            
            if success_bm25 != success_monobert:
                bm25_monobert_different_cases.append((idx, success_no_rag, success_bm25, success_monobert))
            
            if success_no_rag:
                if not success_bm25 or not success_monobert:
                    success_no_rag_but_failed_in_bm25_or_monobert.append((idx, success_no_rag, success_bm25, success_monobert))
    
    
    print(f"n_success_no_rag: {n_success_no_rag}, n_success_bm25: {n_success_bm25}, n_success_monobert: {n_success_monobert}")
    print(f"bm25 and monobert different cases: {bm25_monobert_different_cases}")
    print(f"success_no_rag, but failed in bm25 or monobert: {success_no_rag_but_failed_in_bm25_or_monobert}")
    print(f"success_no_rag_list: {success_no_rag_list}")
    
    print(f"success_no_rag_list true: {list(np.argwhere(np.array(success_no_rag_list) == True).flatten())}")
    print(f"success_no_rag_list false: {list(np.argwhere(np.array(success_no_rag_list) == False).flatten())}")
    return

    # Train with GRPO
    grpo.train(epochs=3)

    # Test GRPO-enhanced reranker
    grpo.rerank_and_test(idx, mode="grpo")


if __name__ == "__main__":
    main()
