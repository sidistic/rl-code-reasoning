"""
Modular Reward Model for RLAIF
Easy to customize and experiment with different reward strategies
"""

import torch
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseRewardModel(ABC):
    """Base class for reward models"""
    
    @abstractmethod
    def compute_reward(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """Compute reward for a solution"""
        pass
    
    def batch_compute(self, prompts: List[str], solutions: List[str], 
                     test_inputs: List[str], expected_outputs: List[str]) -> List[float]:
        """Compute rewards for batch"""
        return [
            self.compute_reward(p, s, ti, eo) 
            for p, s, ti, eo in zip(prompts, solutions, test_inputs, expected_outputs)
        ]

class CodeExecutionReward(BaseRewardModel):
    """Reward based on code execution and correctness"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with customizable weights"""
        self.weights = weights or {
            "syntax": 0.2,      # Valid Python syntax
            "execution": 0.3,   # Runs without errors
            "correctness": 0.4, # Produces correct output
            "style": 0.1        # Code quality/style
        }
    
    def compute_reward(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """Compute reward based on execution and correctness"""
        scores = {}
        
        # 1. Syntax score
        scores["syntax"] = self._check_syntax(solution)
        
        # 2. Execution and correctness
        exec_score, correct_score = self._execute_and_check(solution, test_input, expected_output)
        scores["execution"] = exec_score
        scores["correctness"] = correct_score
        
        # 3. Style score
        scores["style"] = self._check_style(solution)
        
        # Weighted sum
        total = sum(scores[k] * self.weights[k] for k in scores)
        return max(0.0, min(1.0, total))
    
    def _check_syntax(self, code: str) -> float:
        """Check if code has valid syntax"""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
    
    def _execute_and_check(self, code: str, test_input: str, expected_output: str) -> Tuple[float, float]:
        """Execute code and check correctness"""
        try:
            # Extract function name
            func_match = re.search(r'def\s+(\w+)', code)
            if not func_match:
                return 0.0, 0.0
            
            # Execute code
            exec_globals = {}
            exec(code, exec_globals)
            
            # Run test
            result = eval(test_input, exec_globals)
            expected = eval(expected_output) if isinstance(expected_output, str) else expected_output
            
            # Execution succeeded
            exec_score = 1.0
            
            # Check correctness
            correct_score = 1.0 if result == expected else 0.0
            
            return exec_score, correct_score
            
        except Exception as e:
            return 0.0, 0.0
    
    def _check_style(self, code: str) -> float:
        """Basic code style checks"""
        score = 0.0
        
        # Has function definition
        if "def " in code:
            score += 0.3
        
        # Has return statement
        if "return" in code:
            score += 0.3
        
        # Reasonable length
        lines = [l for l in code.split('\n') if l.strip()]
        if 1 <= len(lines) <= 20:
            score += 0.2
        
        # Not just print
        if "def" in code and not code.strip().startswith("print"):
            score += 0.2
        
        return score

class AIFeedbackReward(BaseRewardModel):
    """Reward using AI model feedback (simplified)"""
    
    def __init__(self, model_name: str = None):
        """Initialize with AI model for feedback"""
        # In production, load an actual model
        # For simplicity, using heuristics
        self.model_name = model_name or "mock-ai-model"
    
    def compute_reward(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """Use AI to evaluate solution quality"""
        # Simplified AI evaluation
        score = 0.0
        
        # Check if solution addresses the prompt
        prompt_keywords = re.findall(r'\b\w+\b', prompt.lower())
        solution_lower = solution.lower()
        
        keyword_matches = sum(1 for kw in prompt_keywords if kw in solution_lower)
        score += min(0.3, keyword_matches * 0.05)
        
        # Check structure
        if "def" in solution and "return" in solution:
            score += 0.4
        
        # Length appropriateness
        if 10 < len(solution) < 500:
            score += 0.3
        
        return score

class HybridReward(BaseRewardModel):
    """Combine multiple reward models"""
    
    def __init__(self, models: List[Tuple[BaseRewardModel, float]]):
        """Initialize with list of (model, weight) tuples"""
        self.models = models
        total_weight = sum(w for _, w in models)
        self.models = [(m, w/total_weight) for m, w in models]  # Normalize
    
    def compute_reward(self, prompt: str, solution: str, test_input: str, expected_output: str) -> float:
        """Compute weighted average of multiple rewards"""
        total = 0.0
        for model, weight in self.models:
            reward = model.compute_reward(prompt, solution, test_input, expected_output)
            total += reward * weight
        return total

# Factory function for easy model creation
def create_reward_model(model_type: str = "execution", **kwargs) -> BaseRewardModel:
    """Create reward model by type"""
    if model_type == "execution":
        return CodeExecutionReward(**kwargs)
    elif model_type == "ai":
        return AIFeedbackReward(**kwargs)
    elif model_type == "hybrid":
        # Default hybrid model
        return HybridReward([
            (CodeExecutionReward(), 0.7),
            (AIFeedbackReward(), 0.3)
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Reward Models\n")
    
    # Test data
    prompt = "Write a Python function that adds two numbers"
    good_solution = "def add(a, b):\n    return a + b"
    bad_solution = "print('hello')"
    test_input = "add(2, 3)"
    expected = "5"
    
    # Test different reward models
    models = {
        "Execution": CodeExecutionReward(),
        "AI Feedback": AIFeedbackReward(),
        "Hybrid": create_reward_model("hybrid")
    }
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        good_reward = model.compute_reward(prompt, good_solution, test_input, expected)
        bad_reward = model.compute_reward(prompt, bad_solution, test_input, expected)
        print(f"  Good solution: {good_reward:.3f}")
        print(f"  Bad solution: {bad_reward:.3f}")
    
    # Custom weights example
    print("\n\nCustom Execution Reward (syntax-heavy):")
    custom_model = CodeExecutionReward(weights={
        "syntax": 0.5,
        "execution": 0.2,
        "correctness": 0.2,
        "style": 0.1
    })
    reward = custom_model.compute_reward(prompt, good_solution, test_input, expected)
    print(f"  Reward: {reward:.3f}")