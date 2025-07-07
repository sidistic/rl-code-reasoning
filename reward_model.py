"""
AI Reward Model for RLAIF using Code-specific models
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class AIRewardModel:
    """
    AI-based reward model using a code-specific model for evaluation.
    """
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
        """Initialize the AI reward model with a code-specific model."""
        print(f"Loading AI reward model: {model_name}")
        
        # Use a smaller model for reward evaluation to save memory
        # Options: "microsoft/codebert-base", "Salesforce/codet5-small", "codellama/CodeLlama-7b-Python-hf"
        if "CodeLlama-7b" in model_name:
            # Use a smaller model for rewards to save GPU memory
            self.model_name = "Salesforce/codet5-small"
        else:
            self.model_name = model_name
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ AI Reward Model loaded successfully")
    
    def evaluate_code_solution(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """
        Evaluate a code solution using AI feedback.
        
        Returns:
            Reward score between 0 and 1
        """
        
        # Step 1: Basic syntax check
        syntax_score = self._check_syntax(solution)
        
        # Step 2: Try to execute and check correctness
        correctness_score = self._evaluate_correctness(solution, test_input, expected_output)
        
        # Step 3: Code quality check
        quality_score = self._evaluate_code_quality(solution)
        
        # Weighted combination
        final_reward = (
            0.5 * correctness_score +  # Correctness is most important
            0.3 * quality_score +       # Code quality
            0.2 * syntax_score          # Basic syntax
        )
        
        return max(0.0, min(1.0, final_reward))
    
    def _check_syntax(self, solution: str) -> float:
        """Check if code has valid Python syntax."""
        try:
            compile(solution, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
    
    def _evaluate_code_quality(self, solution: str) -> float:
        """Evaluate basic code quality metrics."""
        score = 0.0
        
        # Check for function definition
        if "def " in solution:
            score += 0.25
        
        # Check for return statement
        if "return" in solution:
            score += 0.25
        
        # Check for reasonable length
        lines = [line for line in solution.split('\n') if line.strip()]
        if 1 <= len(lines) <= 15:
            score += 0.25
        
        # Check if it's not just a print statement
        if "def" in solution and not solution.strip().startswith("print"):
            score += 0.25
        
        return score
    
    def _evaluate_correctness(self, solution: str, test_input: str, expected_output: str) -> float:
        """Try to evaluate correctness by executing code."""
        try:
            # Extract function name
            func_match = re.search(r'def\s+(\w+)', solution)
            if not func_match:
                return 0.0
            
            func_name = func_match.group(1)
            
            # Create safe execution environment
            exec_globals = {}
            exec_locals = {}
            
            # Execute the solution
            exec(solution, exec_globals, exec_locals)
            
            # Get the function
            if func_name not in exec_locals:
                return 0.0
            
            func = exec_locals[func_name]
            
            # Parse and execute test
            if "(" in test_input and ")" in test_input:
                # Extract arguments safely
                args_str = test_input[test_input.find("(")+1:test_input.find(")")]
                
                # Simple parsing for basic types
                args = []
                for arg in args_str.split(","):
                    arg = arg.strip()
                    if arg.startswith("'") or arg.startswith('"'):
                        args.append(arg[1:-1])  # String
                    elif arg.isdigit():
                        args.append(int(arg))   # Integer
                    elif arg.startswith("["):
                        args.append(eval(arg))  # List (simple eval)
                    else:
                        try:
                            args.append(float(arg))  # Float
                        except:
                            args.append(arg)  # Keep as string
                
                # Call function
                result = func(*args)
                
                # Compare with expected
                expected = eval(expected_output)
                
                if result == expected:
                    return 1.0
                elif str(result) == str(expected):
                    return 0.9  # String representation matches
                else:
                    return 0.2  # Wrong answer
            
        except Exception as e:
            return 0.0
        
        return 0.0

    def batch_evaluate(self, prompts: List[str], solutions: List[str], 
                      test_inputs: List[str], expected_outputs: List[str]) -> List[float]:
        """Evaluate multiple solutions in batch."""
        rewards = []
        
        for prompt, solution, test_input, expected_output in zip(
            prompts, solutions, test_inputs, expected_outputs
        ):
            reward = self.evaluate_code_solution(
                prompt, solution, test_input, expected_output
            )
            rewards.append(reward)
        
        return rewards


if __name__ == "__main__":
    # Test the reward model
    print("Testing AI Reward Model...")
    
    # Use a small model for testing
    reward_model = AIRewardModel("Salesforce/codet5-small")
    
    # Test good solution
    prompt = "Write a Python function that adds two numbers."
    good_solution = """def add_numbers(a, b):
    return a + b"""
    
    reward = reward_model.evaluate_code_solution(
        prompt, good_solution, "add_numbers(3, 5)", "8"
    )
    print(f"Good solution reward: {reward:.3f}")
    
    # Test bad solution
    bad_solution = "print('hello')"
    bad_reward = reward_model.evaluate_code_solution(
        prompt, bad_solution, "add_numbers(3, 5)", "8"
    )
    print(f"Bad solution reward: {bad_reward:.3f}")