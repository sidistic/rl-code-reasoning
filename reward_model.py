"""
AI Reward Model for RLAIF (Reinforcement Learning from AI Feedback).
This is the core of RLAIF - using an AI model to generate rewards instead of human feedback.
"""

import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class AIRewardModel:
    """
    AI-based reward model that evaluates code solutions.
    This replaces human feedback in traditional RLHF.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the AI reward model."""
        print(f"Loading AI reward model: {model_name}")
        
        # Use a simple model for reward evaluation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ AI Reward Model loaded successfully")
    
    def evaluate_code_solution(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """
        Evaluate a code solution using AI feedback.
        This is the key RLAIF component.
        
        Args:
            prompt: The original coding problem
            solution: Generated code solution
            test_input: Test case input
            expected_output: Expected result
            
        Returns:
            Reward score between 0 and 1
        """
        
        # Step 1: Check if solution contains code
        code_quality_score = self._evaluate_code_quality(solution)
        
        # Step 2: Try to execute and check correctness
        correctness_score = self._evaluate_correctness(solution, test_input, expected_output)
        
        # Step 3: Use AI to evaluate overall solution quality
        ai_quality_score = self._ai_evaluate_solution(prompt, solution)
        
        # Combine scores (you can adjust weights)
        final_reward = (
            0.4 * correctness_score +  # Correctness is most important
            0.3 * code_quality_score + # Code quality matters
            0.3 * ai_quality_score     # AI judgment
        )
        
        return max(0.0, min(1.0, final_reward))  # Clamp between 0 and 1
    
    def _evaluate_code_quality(self, solution: str) -> float:
        """Evaluate basic code quality."""
        score = 0.0
        
        # Check for function definition
        if "def " in solution:
            score += 0.3
        
        # Check for return statement
        if "return" in solution:
            score += 0.3
        
        # Check for reasonable length (not too short or long)
        lines = [line.strip() for line in solution.split('\n') if line.strip()]
        if 2 <= len(lines) <= 10:
            score += 0.2
        
        # Check for basic Python syntax patterns
        if any(keyword in solution for keyword in ['if', 'for', 'while', '==', '+']):
            score += 0.2
        
        return score
    
    def _evaluate_correctness(self, solution: str, test_input: str, expected_output: str) -> float:
        """Try to evaluate correctness by executing code."""
        try:
            # Extract function name from solution
            func_match = re.search(r'def\s+(\w+)', solution)
            if not func_match:
                return 0.1  # No function found
            
            func_name = func_match.group(1)
            
            # Create safe execution environment
            exec_globals = {"__builtins__": {}}
            exec_locals = {}
            
            # Execute the solution
            exec(solution, exec_globals, exec_locals)
            
            # Get the function
            if func_name not in exec_locals:
                return 0.1  # Function not found
            
            func = exec_locals[func_name]
            
            # Parse test input to extract arguments
            # Simple parsing for basic cases like "add_numbers(3, 5)"
            if "(" in test_input and ")" in test_input:
                args_str = test_input[test_input.find("(")+1:test_input.find(")")]
                args = eval(f"[{args_str}]")  # Simple eval for demo
                
                # Call function
                result = func(*args)
                
                # Compare with expected output
                expected = eval(expected_output)
                
                if result == expected:
                    return 1.0  # Perfect match
                else:
                    return 0.3  # Wrong answer but function works
            
        except Exception as e:
            # Code has errors
            return 0.1
        
        return 0.5  # Default score
    
    def _ai_evaluate_solution(self, prompt: str, solution: str) -> float:
        """Use AI model to evaluate solution quality."""
        
        # Create evaluation prompt for the AI
        eval_prompt = f"""
Rate this code solution on a scale of 0.0 to 1.0:

Problem: {prompt}

Solution:
{solution}

Consider: correctness, clarity, efficiency.
Rating (0.0-1.0):"""

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(eval_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=20,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Extract numerical score
            numbers = re.findall(r'0?\.\d+|[01]\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
        
        except Exception as e:
            print(f"AI evaluation error: {e}")
        
        return 0.5  # Default neutral score

    def batch_evaluate(self, prompts: List[str], solutions: List[str], 
                      test_inputs: List[str], expected_outputs: List[str]) -> List[float]:
        """Evaluate multiple solutions in batch."""
        rewards = []
        
        for prompt, solution, test_input, expected_output in zip(prompts, solutions, test_inputs, expected_outputs):
            reward = self.evaluate_code_solution(prompt, solution, test_input, expected_output)
            rewards.append(reward)
        
        return rewards

# Create reward function for TRL integration
def create_reward_function(reward_model: AIRewardModel):
    """Create a reward function compatible with TRL trainers."""
    
    def reward_fn(samples: Dict[str, Any]) -> List[float]:
        """
        Reward function for RLAIF training.
        
        Args:
            samples: Dictionary with 'query' and 'response' keys
            
        Returns:
            List of reward scores
        """
        queries = samples.get('query', [])
        responses = samples.get('response', [])
        
        # For this simple demo, we'll use dummy test cases
        # In practice, you'd extract these from the query or have them stored
        test_inputs = ["dummy_test()" for _ in queries]
        expected_outputs = ["dummy_result" for _ in queries]
        
        rewards = reward_model.batch_evaluate(queries, responses, test_inputs, expected_outputs)
        
        print(f"Generated {len(rewards)} rewards: avg={sum(rewards)/len(rewards):.3f}")
        
        return rewards
    
    return reward_fn

if __name__ == "__main__":
    # Test the reward model
    print("Testing AI Reward Model...")
    
    reward_model = AIRewardModel()
    
    # Test case
    prompt = "Write a Python function that adds two numbers."
    solution = """def add_numbers(a, b):
    return a + b"""
    test_input = "add_numbers(3, 5)"
    expected_output = "8"
    
    reward = reward_model.evaluate_code_solution(prompt, solution, test_input, expected_output)
    print(f"Reward for solution: {reward:.3f}")
    
    # Test bad solution
    bad_solution = "print('hello')"
    bad_reward = reward_model.evaluate_code_solution(prompt, bad_solution, test_input, expected_output)
    print(f"Reward for bad solution: {bad_reward:.3f}")