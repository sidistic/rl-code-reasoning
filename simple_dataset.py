"""
Simple dataset creation for RLAIF code generation.
Creates basic programming problems for learning RLAIF concepts.
"""

import json
from datasets import Dataset

def create_simple_coding_problems():
    """Create a set of simple coding problems for RLAIF training."""
    
    problems = [
        {
            "prompt": "Write a Python function that adds two numbers.",
            "test_input": "add_numbers(3, 5)",
            "expected_output": "8",
            "difficulty": "easy"
        },
        {
            "prompt": "Write a Python function that finds the maximum of two numbers.",
            "test_input": "max_of_two(10, 7)",
            "expected_output": "10", 
            "difficulty": "easy"
        },
        {
            "prompt": "Write a Python function that checks if a number is even.",
            "test_input": "is_even(4)",
            "expected_output": "True",
            "difficulty": "easy"
        },
        {
            "prompt": "Write a Python function that reverses a string.",
            "test_input": "reverse_string('hello')",
            "expected_output": "'olleh'",
            "difficulty": "medium"
        },
        {
            "prompt": "Write a Python function that finds the factorial of a number.",
            "test_input": "factorial(5)",
            "expected_output": "120",
            "difficulty": "medium"
        },
        {
            "prompt": "Write a Python function that checks if a string is a palindrome.",
            "test_input": "is_palindrome('racecar')",
            "expected_output": "True",
            "difficulty": "medium"
        },
        {
            "prompt": "Write a Python function that finds the sum of all elements in a list.",
            "test_input": "sum_list([1, 2, 3, 4, 5])",
            "expected_output": "15",
            "difficulty": "easy"
        },
        {
            "prompt": "Write a Python function that counts vowels in a string.",
            "test_input": "count_vowels('hello world')",
            "expected_output": "3",
            "difficulty": "medium"
        }
    ]
    
    return problems

def format_for_training(problems):
    """Format problems for RLAIF training."""
    formatted = []
    
    for problem in problems:
        # Create instruction-following format
        instruction = f"{problem['prompt']}\n\nExample:\n{problem['test_input']} should return {problem['expected_output']}"
        
        formatted.append({
            "query": instruction,
            "test_input": problem["test_input"], 
            "expected_output": problem["expected_output"],
            "difficulty": problem["difficulty"]
        })
    
    return formatted

def create_dataset():
    """Create and return the training dataset."""
    problems = create_simple_coding_problems()
    formatted_problems = format_for_training(problems)
    
    # Create train/test split
    train_data = formatted_problems[:6]  # First 6 for training
    test_data = formatted_problems[6:]   # Last 2 for testing
    
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    print(f"Created {len(train_data)} training examples and {len(test_data)} test examples")
    
    return train_dataset, test_dataset

def save_dataset():
    """Save dataset to files for inspection."""
    train_dataset, test_dataset = create_dataset()
    
    # Save as JSON for easy viewing
    with open("train_data.json", "w") as f:
        json.dump(train_dataset.to_list(), f, indent=2)
    
    with open("test_data.json", "w") as f:
        json.dump(test_dataset.to_list(), f, indent=2)
    
    print("Datasets saved to train_data.json and test_data.json")
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    save_dataset()