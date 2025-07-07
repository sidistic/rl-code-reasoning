"""
Model Evaluation Script
Test your RLAIF-trained model on coding problems
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from typing import Dict, List

class ModelEvaluator:
    """Evaluate RLAIF-trained models"""
    
    def __init__(self, model_path: str):
        """Load trained model"""
        print(f"📦 Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_length: int = 150) -> str:
        """Generate code for a prompt"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only generated part
        generated = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    def execute_code(self, code: str, test_input: str) -> tuple:
        """Safely execute generated code"""
        try:
            # Extract function name
            import re
            func_match = re.search(r'def\s+(\w+)', code)
            if not func_match:
                return False, "No function definition found"
            
            func_name = func_match.group(1)
            
            # Execute code
            exec_globals = {}
            exec(code, exec_globals)
            
            # Get function
            if func_name not in exec_globals:
                return False, "Function not found after execution"
            
            # Parse test input and execute
            # Simple parsing: extract function call
            if "(" in test_input and ")" in test_input:
                # Build the call
                result = eval(test_input, exec_globals)
                return True, result
            
        except Exception as e:
            return False, str(e)
        
        return False, "Execution failed"
    
    def evaluate_single(self, prompt: str, test_cases: List[Dict]) -> Dict:
        """Evaluate model on single problem"""
        # Generate solution
        generated = self.generate(prompt)
        
        # Test the solution
        passed = 0
        total = len(test_cases)
        details = []
        
        for test in test_cases:
            success, result = self.execute_code(generated, test["input"])
            
            if success:
                expected = eval(test["output"]) if isinstance(test["output"], str) else test["output"]
                if result == expected:
                    passed += 1
                    details.append(f"✅ {test['input']} → {result}")
                else:
                    details.append(f"❌ {test['input']} → {result} (expected {expected})")
            else:
                details.append(f"❌ {test['input']} → Error: {result}")
        
        return {
            "prompt": prompt,
            "generated": generated,
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0,
            "details": details
        }
    
    def evaluate_dataset(self, test_file: str = "leetcode_problems.json") -> Dict:
        """Evaluate on full dataset"""
        # Load test data
        with open(test_file, 'r') as f:
            problems = json.load(f)
        
        print(f"\n🧪 Evaluating on {len(problems)} problems...")
        
        results = []
        total_passed = 0
        total_tests = 0
        
        for i, problem in enumerate(problems[:10]):  # Limit to 10 for speed
            print(f"\n📝 Problem {i+1}: {problem['prompt'][:60]}...")
            
            result = self.evaluate_single(problem["prompt"], problem["test_cases"])
            results.append(result)
            
            total_passed += result["passed"]
            total_tests += result["total"]
            
            print(f"Generated: {result['generated'][:80]}...")
            print(f"Tests: {result['passed']}/{result['total']} passed")
        
        # Summary
        overall_success = total_passed / total_tests if total_tests > 0 else 0
        
        return {
            "total_problems": len(results),
            "total_tests_passed": total_passed,
            "total_tests": total_tests,
            "overall_success_rate": overall_success,
            "results": results
        }
    
    def interactive_mode(self):
        """Interactive testing"""
        print("\n🎮 Interactive Mode - Type 'quit' to exit")
        print("Enter coding problems to see generated solutions\n")
        
        while True:
            prompt = input("💡 Problem: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            print("🤖 Generating...")
            solution = self.generate(prompt)
            print(f"\n📝 Solution:\n{solution}\n")
            
            # Optional: test with user input
            test = input("🧪 Test case (e.g., 'func(1,2)' or skip): ")
            if test and test.strip():
                success, result = self.execute_code(solution, test)
                if success:
                    print(f"✅ Result: {result}")
                else:
                    print(f"❌ Error: {result}")
            
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RLAIF model")
    parser.add_argument("--model_path", default="./trained_model",
                       help="Path to trained model")
    parser.add_argument("--test_file", default="leetcode_problems.json",
                       help="Test problems file")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive testing mode")
    parser.add_argument("--save_results", help="Save results to file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(args.model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you've trained a model first!")
        return
    
    if args.interactive:
        evaluator.interactive_mode()
    else:
        # Run evaluation
        results = evaluator.evaluate_dataset(args.test_file)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"Problems tested: {results['total_problems']}")
        print(f"Total tests passed: {results['total_tests_passed']}/{results['total_tests']}")
        print(f"Overall success rate: {results['overall_success_rate']:.1%}")
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.save_results}")

if __name__ == "__main__":
    main()