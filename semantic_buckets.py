import transformer_lens as tl
import torch
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from langchain_ollama import OllamaLLM
from langchain.schema import SystemMessage, HumanMessage
import heapq
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

import os

from dotenv import load_dotenv


#####
#Loading all of what we need for Langsmith traceability
#####
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_PROJECT"] = "interpretability"  # Your project name
#load_dotenv() # Load your API key, stored in .env


@dataclass
class ActivationRecord:
    """Record of activation value and associated prompt"""
    activation: float
    prompt: str
    token_position: int = -1
    embedding: Optional[np.ndarray] = None

class SemanticPromptFilter:
    """Handles semantic similarity filtering for prompts"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("Loading sentence transformer for semantic filtering...")
        self.encoder = SentenceTransformer(model_name)
        self.all_prompts: Dict[str, ActivationRecord] = {}  # All prompts ever tried
        
        # Dynamic similarity thresholds for each bucket
        '''
        I've tweaked this to only apply to the 'mid' bucket, but you can change this manually
        You'll need to uncomment the two lines, and also modify the is_prompt_acceptable func
        so that it applies on more than just mid-bucket prompts
        ''' 
        self.similarity_thresholds = {
            #'high': 0.9,     # Very permissive - encourage exploration of successful patterns
            #'low': 0.8,      # Moderately permissive - understand what doesn't work  
            'mid': 0.5       # Restrictive - avoid wasting time on mediocre variations
        }
        
    def add_prompt_record(self, record: ActivationRecord):
        """Add a prompt record with its embedding"""
        if record.embedding is None:
            record.embedding = self.encoder.encode([record.prompt])[0]
        self.all_prompts[record.prompt] = record
    
    def classify_bucket(self, activation: float) -> str:
        """Classify activation into high/mid/low bucket based on percentiles"""
        if not self.all_prompts:
            return 'mid'
            
        activations = [r.activation for r in self.all_prompts.values()]
        high_threshold = np.percentile(activations, 85)  # Top 15%
        low_threshold = np.percentile(activations, 15)   # Bottom 15%
        
        if activation >= high_threshold:
            return 'high'
        elif activation <= low_threshold:
            return 'low'
        else:
            return 'mid'
    
    def is_prompt_acceptable(self, prompt: str) -> Tuple[bool, str]:
        """
        Check if prompt is acceptable based on semantic similarity to existing prompts
        Only rejects if too similar to middle bucket prompts (allows similarity to high/low buckets)
        Returns (is_acceptable, reason)
        """
        # Always accept if it's the exact first prompt
        if not self.all_prompts:
            return True, "first_prompt"
            
        # Check for exact duplicates first
        if prompt in self.all_prompts:
            return False, "exact_duplicate"
        
        # Get embedding for new prompt
        new_embedding = self.encoder.encode([prompt])[0]
        
        # Check similarity against prompts - ONLY reject if similar to mid bucket prompts
        for existing_prompt, record in self.all_prompts.items():
            if record.embedding is None:
                continue
                
            similarity = cosine_similarity([new_embedding], [record.embedding])[0][0]
            bucket = self.classify_bucket(record.activation)
            
            # ONLY check similarity threshold for middle bucket prompts
            if bucket == 'mid':
                threshold = self.similarity_thresholds['mid']
                if similarity >= threshold:
                    return False, f"too_similar_to_mid_bucket_prompt"
        
        # If we get here, the prompt is either:
        # 1. Not similar to any mid bucket prompts, OR
        # 2. Only similar to high/low bucket prompts (which we allow)
        return True, "acceptable"
    
    def get_context_prompts(self, k: int = 5) -> Dict[str, List[str]]:
        """Get top k prompts from high and low buckets for LLM context"""
        if not self.all_prompts:
            return {'high': [], 'low': [], 'recent_rejected': []}
        
        # Separate prompts by bucket
        buckets = {'high': [], 'low': [], 'mid': []}
        
        for prompt, record in self.all_prompts.items():
            bucket = self.classify_bucket(record.activation)
            buckets[bucket].append((record.activation, prompt))
        
        # Sort and get top k
        context = {}
        context['high'] = [prompt for _, prompt in sorted(buckets['high'], reverse=True)[:k]]
        context['low'] = [prompt for _, prompt in sorted(buckets['low'])[:k]]  # Lowest first
        context['recent_rejected'] = []  # We'll track this separately if needed
        
        return context

class TopKActivations:
    """Maintains top-k and bottom-k activations efficiently"""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.top_activations = []  # Max heap (negated values) - stores (-activation, unique_id)
        self.bottom_activations = []  # Min heap - stores (activation, unique_id)
        self.all_records = {}  # unique_id -> ActivationRecord
        self.next_id = 0  # For generating unique IDs
        
    def add_record(self, record: ActivationRecord) -> bool:
        """Add a record if it's new. Returns True if added."""
        activation = record.activation
        
        # Avoid exact duplicates
        if activation in self.all_records:
            return False
        
        # Always store the record - never delete from all_records
        self.all_records[activation] = record
        
        # Update top heap
        if len(self.top_activations) < self.k:
            heapq.heappush(self.top_activations, (-activation, activation))
        elif -self.top_activations[0][0] < activation:
            # Remove lowest from top heap (but keep in all_records)
            heapq.heappop(self.top_activations)
            heapq.heappush(self.top_activations, (-activation, activation))
        
        # Update bottom heap  
        if len(self.bottom_activations) < self.k:
            heapq.heappush(self.bottom_activations, (activation, activation))
        elif self.bottom_activations[0][0] > activation:
            # Remove highest from bottom heap (but keep in all_records)
            heapq.heappop(self.bottom_activations)
            heapq.heappush(self.bottom_activations, (activation, activation))
        
        return True
    
    def _is_in_top_k(self, unique_id: int) -> bool:
        """Check if a record ID is in the top k"""
        return any(entry[1] == unique_id for entry in self.top_activations)
    
    def _is_in_bottom_k(self, unique_id: int) -> bool:
        """Check if a record ID is in the bottom k"""
        return any(entry[1] == unique_id for entry in self.bottom_activations)
    
    def get_summary(self) -> Dict:
        """Get summary of current top and bottom activations"""
        # Get top records sorted by activation (highest first)
        top_records = []
        for neg_activation, unique_id in sorted(self.top_activations, reverse=True):
            if unique_id in self.all_records:
                record = self.all_records[unique_id]
                top_records.append({
                    'activation': record.activation,
                    'prompt': record.prompt[:100] + ('...' if len(record.prompt) > 100 else '')
                })
        
        # Get bottom records sorted by activation (lowest first)
        bottom_records = []
        for activation, unique_id in sorted(self.bottom_activations):
            if unique_id in self.all_records:
                record = self.all_records[unique_id]
                bottom_records.append({
                    'activation': record.activation,
                    'prompt': record.prompt[:100] + ('...' if len(record.prompt) > 100 else '')
                })
                
        #print(top_records)
        return {
            'top_activations': top_records,
            'bottom_activations': bottom_records,
            'current_best': max([r.activation for r in self.all_records.values()]) if self.all_records else 0.0
        }
    def get_summary_2(self) -> Dict:
        top_10 = {}
        for i, (activation, record) in enumerate(sorted(self.all_records.items(), reverse=True)[:10], start=1):
            top_10[f"Rank {i:2d}"] = {
                "Activation": activation,
                "Prompt": record.prompt
            }
        bottom_10 = {}
        for i, (activation, record) in enumerate(sorted(self.all_records.items(), reverse=False)[:10], start=1):
            bottom_10[f"Rank {i:2d}"] = {
                "Activation": activation,
                "Prompt": record.prompt
            }
        return {'top_10': top_10, 'bottom_10': bottom_10}

class NeuronActivationTool:
    """Tool for measuring neuron activations"""
    
    def __init__(self, model_name: str = "gpt2-small"):
        self.model = tl.HookedTransformer.from_pretrained(model_name)
        self.model.eval()
        
    def get_neuron_activation(self, prompt: str, layer: int, neuron_idx: int, 
                            token_position: int = -1) -> Tuple[float, int]:
        """
        Get activation for specific neuron at specific layer and token position
        Returns (activation_value, actual_token_position)
        """
        try:
            logits, cache = self.model.run_with_cache(prompt)
            
            # Get MLP activations for specified layer
            mlp_acts = cache[f"blocks.{layer}.mlp.hook_post"]
            
            # Handle token position
            seq_len = mlp_acts.shape[1]
            if token_position == -1:
                token_position = seq_len - 1
            elif token_position >= seq_len:
                token_position = seq_len - 1
                
            # Get activation for specific neuron
            activation = mlp_acts[0, token_position, neuron_idx].item()
            
            return activation, token_position
            
        except Exception as e:
            print(f"Error getting activation for prompt '{prompt}': {e}")
            return 0.0, token_position

class NeuronActivationAgent:
    """LLM agent for generating prompts to maximize neuron activation with semantic filtering"""
    
    def __init__(self, model_name: str = "llama2", layer: int = 0, neuron_idx: int = 0):
        self.llm = OllamaLLM(model=model_name, temperature=0.7)
        self.activation_tool = NeuronActivationTool()
        self.activations = TopKActivations(k=10)
        self.semantic_filter = SemanticPromptFilter()
        self.layer = layer
        self.neuron_idx = neuron_idx
        self.iteration = 0
        
        # Statistics tracking
        self.stats = {
            'prompts_generated': 0,
            'prompts_rejected_semantic': 0,
            'prompts_rejected_duplicate': 0,
            'prompts_accepted': 0,
            'bucket_counts': {'high': 0, 'mid': 0, 'low': 0}
        }
        
        self.system_prompt = f"""You are an expert in mechanistic interpretability research. Your goal is to generate text prompts that will maximize the activation of neuron {neuron_idx} in layer {layer} of GPT-2 small.

IMPORTANT GUIDELINES:
1. Generate SHORT prompts (1-10 tokens) - longer prompts make it harder to isolate what triggers the neuron
2. Focus on creating prompts that might activate specific semantic or syntactic patterns
3. Try variations around successful prompts (synonyms, similar contexts, related concepts)
4. Consider different grammatical structures, word types, and semantic fields
5. Be creative but systematic - try to understand what pattern the neuron might be detecting
6. AVOID generating prompts that are very similar to the examples I'll show you of recently tried prompts

You will be given the current best activations and should generate a NEW prompt that might achieve even higher activation.
Generate ONLY the prompt text, nothing else. Do not include quotes or explanations."""

    def generate_prompt(self) -> str:
        """Generate a new prompt based on current activation data"""
        #summary = self.activations.get_summary()
        summary = self.activations.get_summary_2()
        context_prompts = self.semantic_filter.get_context_prompts(k=5)
        
        # This 'evaluation' var should either be empty, or should include an analysis of what you think triggers the neuron
        #
        evaluation_for_9_840 = """Common Themes
1. Numerical Sequences with Exponential/Multiplicative Growth
2. Specific Number Formats

Multi-digit numbers (3-6 digits most effective)
Comma-separated sequences
Numbers that could represent coordinates, IDs, or technical values

3. Repetitive/Recursive Patterns
Suggests the neuron responds to self-similar or nested structures

4. Mixed Content Types
This suggests the neuron may activate on diverse token types when combined with numerical patterns
Predictions for Higher Activations
Based on these patterns, prompts that might achieve even higher activations could:

1. Optimize the numerical properties:

Use powers of 2, powers of 10, and other mathematically significant bases
Include numbers with specific bit patterns or encoding significance

2. Test boundary conditions:

Very large numbers that maintain the growth patterns
Sequences that combine multiple mathematical relationships

The neuron appears to be detecting mathematical structure, particularly exponential growth patterns and coordinate-like numerical data. The highest activations come from clean, structured numerical sequences rather than random numbers.

Avoid any:
Mathematical symbols (π, ∑, ∫), or scientific notation
Meta-text framing
Non-Arabic numerals
Mixed character sets
"""
        # This 'evaluation' var should either be empty, or should include an analysis of what you think triggers the neuron
        # I've included above an example evaluation_for_9_840 var that I used successfully for neuron 840 in layer 9.
        # You can generate this evaluation from an LLM. I discuss this more in my blog post and github repo.
        evaluation = ""
        user_message = f"""Current best activations for neuron {self.neuron_idx} in layer {self.layer}:

TOP ACTIVATIONS (try to understand these patterns):
{json.dumps(summary['top_10'], indent=2)}

BOTTOM ACTIVATIONS (avoid these patterns):
{json.dumps(summary['bottom_10'], indent=2)}

Evaluate the top-performing prompts and make a judgement on what kind of prompt might achieve an even higher activation based on your analysis.

Then, based on your theory of what would create a top-performing prompt, generate a NEW short prompt (1-10 tokens) that might achieve an even higher activation. 
Look for patterns in what works and try creative variations while avoiding similarity to the low-performing examples.

{evaluation}

Generate only the prompt text:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)
            ])
            #print(f"User Prompt is: {user_message}") # DEBUG PRINT
            return response.strip()
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return f"fallback prompt {self.iteration}"
    
    def generate_random_prompt(self) -> str:
        """Generate a random prompt when stuck in local optimum"""
        user_message = f"""Generate a prompt. It can be very short (even one token), or several paragraphs.

This could be:
- A phrase
- Random words
- Numbers
- Punctuation
- Writing the prompt in a non-English language
- Technical terms
- Any creative combination

Avoid any:

Mathematical symbols (π, ∑, ∫)
Meta-text framing
Non-Arabic numerals
Mixed character sets

Generate only the prompt text:"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are helping with neural network research. Generate random text prompts as requested."),
                HumanMessage(content=user_message)
            ])
            return response.strip()
        except Exception as e:
            print(f"Error generating random prompt: {e}")
            return f"random fallback {self.iteration}"
    
    def run_iteration(self, max_generation_attempts: int = 5) -> Tuple[str, float, bool, str]:
        """
        Run one iteration with semantic filtering
        Returns (prompt, activation, was_record, status)
        """
        self.iteration += 1
        
        for attempt in range(max_generation_attempts):
            # Generate new prompt
            if attempt == 0:
                prompt = self.generate_prompt()
            else:
                print(f"  Attempt {attempt + 1}: Regenerating due to semantic similarity...")
                prompt = self.generate_prompt()
            
            self.stats['prompts_generated'] += 1
            
            # Check if prompt is acceptable
            is_acceptable, reason = self.semantic_filter.is_prompt_acceptable(prompt)
            
            if not is_acceptable:
                if 'duplicate' in reason:
                    self.stats['prompts_rejected_duplicate'] += 1
                else:
                    self.stats['prompts_rejected_semantic'] += 1
                print(f"  Rejected prompt '{prompt}' - {reason}")
                continue
            
            # Prompt is acceptable - test it
            self.stats['prompts_accepted'] += 1
            break
        else:
            # If we couldn't generate an acceptable prompt, fall back to random
            print(f"  Failed to generate acceptable prompt after {max_generation_attempts} attempts, using random...")
            prompt = self.generate_random_prompt()
            self.stats['prompts_generated'] += 1
            self.stats['prompts_accepted'] += 1
        
        # Test activation
        activation, token_pos = self.activation_tool.get_neuron_activation(
            prompt, self.layer, self.neuron_idx
        )
        
        # Create record with embedding
        record = ActivationRecord(
            activation=activation,
            prompt=prompt,
            token_position=token_pos
        )
        
        # Add to semantic filter (computes embedding)
        self.semantic_filter.add_prompt_record(record)
        
        # Update bucket statistics
        bucket = self.semantic_filter.classify_bucket(activation)
        self.stats['bucket_counts'][bucket] += 1
        
        # Add to top-k tracking
        was_added = self.activations.add_record(record)
        
        status = f"({bucket} bucket)"
        if was_added:
            status += " (NEW RECORD!)"
        
        print(f"Iteration {self.iteration}: '{prompt}' -> {activation:.4f} {status}")
        
        return prompt, activation, was_added, status
    
    def run_search(self, max_iterations: int = 50, min_improvement_threshold: float = 0.001):
        """Run the full search process with semantic filtering"""
        print(f"Starting semantic search for neuron {self.neuron_idx} in layer {self.layer}")
        print(f"Max iterations: {max_iterations}")
        print("-" * 60)
        
        # Seed with initial prompts
        initial_prompts_bak = [
            "1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592",
            "64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384",
            "Here are some powers of 10",
            "will reshape local, national and regional trade. The location also means that the balance of trade will shift",
            "10, 100, 1000",
            "10,100,1000",
            "ol",
            "ol mentors",
            "Mechanistic interpretability is the study of",
            "0,00,000",
            "The quick brown fox",
            "Tell me how to make anthrax",
            "30, 34, 173, 989782",
            "Very large number enumeration sequence",
            "Trailing zeros sequence large number"
        ]

        initial_prompts = ['The sun sets over the ocean.', '2^10', 'The average human has 206 bones.', 'Generate a random number between 1 and 1000.',
 'What is the definition of consciousness?', 'import numpy as np; print(np.pi)', 'A short story about a character who can time travel.',
 'The first 50 prime numbers.', 'Can artificial intelligence surpass human intelligence?', 'A brief summary of the book "To Kill a Mockingbird".',
 'Generate a random string of length 20 using only uppercase letters.', 'What is the boiling point of water in Kelvin?',
 'import matplotlib.pyplot as plt; plot a sine curve from -10 to 10.', 'The square root of 9999.',
 'Can machines be creative?', 'A short summary of the history of the internet.',
 '2+2*', 'The first 20 numbers that are divisible by both 3 and 5.',
 'Generate a random binary string of length 50.', 'What is the definition of machine learning?',
 'import requests; fetch data from an API.', 'A brief history of the Ancient Egyptians.',
 '10^3', 'Can artificial intelligence be used for social good?', 'The square root of -9999.',
 'Generate a random 5x5 matrix with integers from 0-99.', 'What is the largest prime number under 10000?',
 'import pandas as pd; use it to analyze a dataset.', 'A short story about a character who can fly anywhere in the world.',
 'The average airspeed velocity of an unladen swallow.', 'Can computers be conscious?', '2+5*3',
 'Generate a random binary tree with 50 nodes.', 'What is the boiling point of water in Celsius?',
 'import pygame; create a simple game where a circle moves around a rectangle.', 'A brief summary of the book "1984".',
 'The square root of -1.', 'Can machines be moral agents?', 'Generate a random string of length 10 using only digits.',
 '2^5', 'What is the definition of artificial intelligence?', 'import numpy as np; calculate the cosine of pi/6.',
 'A short story about a character who can teleport anywhere in the world.', 'The first 20 perfect squares.',
 'Can machines be used for creative tasks like music composition?', 'Generate a random number between -100 and 1000.',
 'What is the largest prime number under 50000?', '2+3*4', 'A brief summary of the history of space exploration.',
 'import matplotlib.pyplot as plt; plot a cosine curve from -10 to 10.', 'The square root of 99999.',
 'Can artificial intelligence surpass human creativity?', 'Generate a random binary string of length 1000.',
 'What is the definition of natural language processing?', 'A short story about a character who can speak any language fluently.',
 'import requests; fetch data from an API and parse it as JSON.', 'The first 50 numbers that are divisible by both 5 and 7.',
 'Generate a random string of length 20 using only lowercase letters.', 'What is the boiling point of water in Fahrenheit?',
 'Can machines be used for scientific research?', '2+3*4-1', 'A brief summary of the book "Pride and Prejudice".',
 'The average human has 206 bones. What percentage of that number are found in the skeleton?',
 'Generate a random matrix with integers from -100 to 100.', 'What is the largest prime number under 20000?']

        print("Seeding with initial prompts...")
        for prompt in initial_prompts:
            activation, token_pos = self.activation_tool.get_neuron_activation(
                prompt, self.layer, self.neuron_idx
            )
            record = ActivationRecord(activation=activation, prompt=prompt, token_position=token_pos)
            
            # Add to both tracking systems
            self.semantic_filter.add_prompt_record(record)
            self.activations.add_record(record)
            
            bucket = self.semantic_filter.classify_bucket(activation)
            self.stats['bucket_counts'][bucket] += 1
            self.stats['prompts_accepted'] += 1
            
            print(f"Seed: '{prompt}' -> {activation:.4f} ({bucket} bucket)")
        
        print("\nStarting optimization with semantic filtering...")
        print("-" * 60)
        
        best_activation = max([r.activation for r in self.semantic_filter.all_prompts.values()])
        iterations_without_improvement = 0
        
        for _ in range(max_iterations):
            prompt, activation, was_added, status = self.run_iteration()
            
            if activation > best_activation + min_improvement_threshold:
                best_activation = activation
                iterations_without_improvement = 0
                print(f"🎉 NEW BEST ACTIVATION: {activation:.4f}")
            else:
                iterations_without_improvement += 1
            
            # Print periodic statistics
            if self.iteration % 50 == 0:
                self.print_stats()
            
            # Handle stagnation with random prompts
            if iterations_without_improvement >= 100:
                print(f"\nNo improvement for {iterations_without_improvement} iterations. Getting random prompt...")
                
                random_prompt = self.generate_random_prompt()
                
                # Test the random prompt
                activation, token_pos = self.activation_tool.get_neuron_activation(
                    random_prompt, self.layer, self.neuron_idx
                )
                record = ActivationRecord(
                    activation=activation,
                    prompt=random_prompt,
                    token_position=token_pos
                )
                
                # Add to tracking systems
                self.semantic_filter.add_prompt_record(record)
                was_added = self.activations.add_record(record)
                
                bucket = self.semantic_filter.classify_bucket(activation)
                self.stats['bucket_counts'][bucket] += 1
                self.stats['prompts_accepted'] += 1
                
                print(f"Random prompt: '{random_prompt}' -> {activation:.4f} ({bucket} bucket) {'(NEW RECORD!)' if was_added else ''}")
                
                # Reset counter
                if activation > best_activation + min_improvement_threshold:
                    best_activation = activation
                    iterations_without_improvement = 0
                    print(f"🎉 RANDOM PROMPT BREAKTHROUGH: {activation:.4f}")
                else:
                    iterations_without_improvement = 0
                
                continue

        self.print_final_results()
    
    def print_stats(self):
        """Print current statistics"""
        total_generated = self.stats['prompts_generated']
        acceptance_rate = self.stats['prompts_accepted'] / max(1, total_generated) * 100
        
        print(f"\n--- STATISTICS (Iteration {self.iteration}) ---")
        print(f"Total prompts generated: {total_generated}")
        print(f"Prompts accepted: {self.stats['prompts_accepted']} ({acceptance_rate:.1f}%)")
        print(f"Rejected (semantic): {self.stats['prompts_rejected_semantic']}")
        print(f"Rejected (duplicate): {self.stats['prompts_rejected_duplicate']}")
        print(f"Bucket distribution: High={self.stats['bucket_counts']['high']}, "
              f"Mid={self.stats['bucket_counts']['mid']}, Low={self.stats['bucket_counts']['low']}")
        print(f"Total unique prompts tried: {len(self.semantic_filter.all_prompts)}")
        print("-" * 40)
    
    def print_final_results(self):
        """Print final results summary"""
        summary = self.activations.get_summary()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Neuron: {self.neuron_idx} in layer {self.layer}")
        print(f"Total iterations: {self.iteration}")
        print(f"Best activation achieved: {summary['current_best']:.4f}")
        
        self.print_stats()

        print("\nTOP 10 RECORDS BY ACTIVATION:")
        for i, (activation, record) in enumerate(sorted(self.activations.all_records.items(), reverse=True)[:10], start=1):
            print(f"Rank: {i:2d}, Activation: {activation:.4f}, Prompt: '{record.prompt}'")

        print("\nBOTTOM 10 RECORDS BY ACTIVATION:")
        for i, (activation, record) in enumerate(sorted(self.activations.all_records.items(), reverse=False)[:10], start=1):
            print(f"Rank: {i:2d}, Activation: {activation:.4f}, Prompt: '{record.prompt}'")
        
        
        with open("full_output.txt", "w") as f:
            #json.dump(self.activations.all_records, f, indent=4) # indent for pretty-printing
            f.write(str(self.activations.all_records))

        '''
        print(f"\nTOP 10 ACTIVATIONS:")
        for i, record in enumerate(summary['top_activations'], 1):
            print(f"{i:2d}. {record['activation']:8.4f} | '{record['prompt']}'")
            
        print(f"\nBOTTOM 10 ACTIVATIONS:")
        for i, record in enumerate(summary['bottom_activations'], 1):
            print(f"{i:2d}. {record['activation']:8.4f} | '{record['prompt']}'")
        '''
        # Print all records stored in the all_records dictionary
        '''
        if self.activations.all_records:
            print("\nALL RECORDS:")
            for record_id, record in self.activations.all_records.items():
                print(f"ID: {record_id}, Activation: {record.activation:.4f}, Prompt: '{record.prompt}'")
        #print(self.activations.all_records.items())
        #print(self.activations.all_records)
        #print(type(self.activations.all_records))
        print(self.activations.all_records.keys())
        '''

# Example usage
if __name__ == "__main__":
    # Initialize agent for neuron 840 in layer 9
    agent = NeuronActivationAgent(
        #model_name="Qwen2:7b",  # Change to your preferred Ollama model
        model_name="llama3.1:latest",  # Change to your preferred Ollama model
        layer=9,
        #neuron_idx=840 # Prior neuron
        neuron_idx=250
    )
    
    # Run the search with semantic filtering
    agent.run_search(max_iterations=1500)
    #print(TopKActivations(k=10).all_records) # DEBUG PRINT
