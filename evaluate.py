"""
Simple evaluation script for RLAIF trained models.
Test your trained model on coding problems.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from simple_dataset import create_dataset
from reward_model import AIRewardModel
import argparse

def load_trained_model(model_path: str):
    """Load a trained model."""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_solution(model, tokenizer, prompt: str) -> str:
    """Generate a code solution for a given prompt."""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return generated.strip()

def evaluate_model(model_path: str = "./trained_model"):
    """Evaluate a trained model."""
    
    # Load model
    try:
        model, tokenizer = load_trained_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you have trained a model first with: python train.py")
        return
    
    # Load test data
    _, test_dataset = create_dataset()
    
    # Load reward model for evaluation
    reward_model = AIRewardModel()
    
    print(f"\n🧪 Testing on {len(test_dataset)} problems...")
    print("=" * 60)
    
    total_reward = 0.0
    successful_solutions = 0
    
    for i, problem in enumerate(test_dataset):
        print(f"\n📝 Problem {i+1}: {problem['query'][:50]}...")
        
        # Generate solution
        solution = generate_solution(model, tokenizer, problem['query'])
        print(f"🤖 Generated: {solution[:100]}...")
        
        # Evaluate with AI reward model
        reward = reward_model.evaluate_code_solution(
            prompt=problem['query'],
            solution=solution,
            test_input=problem['test_input'],
            expected_output=problem['expected_output']
        )
        
        total_reward += reward
        if reward > 0.6:  # Consider >0.6 as successful
            successful_solutions += 1
            print(f"✅ Reward: {reward:.3f} (Good!)")
        else:
            print(f"⚠️  Reward: {reward:.3f} (Needs improvement)")
    
    # Summary
    avg_reward = total_reward / len(test_dataset)
    success_rate = successful_solutions / len(test_dataset)
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Successful Solutions: {successful_solutions}/{len(test_dataset)}")
    
    if avg_reward > 0.6:
        print("🎉 Great! Your model is performing well!")
    elif avg_reward > 0.4:
        print("👍 Not bad! Try training for more episodes.")
    else:
        print("🔄 Model needs more training. Try increasing episodes.")

def interactive_test(model_path: str = "./trained_model"):
    """Interactive testing - type your own problems."""
    
    try:
        model, tokenizer = load_trained_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    reward_model = AIRewardModel()
    
    print("\n🎮 Interactive Testing Mode")
    print("Type coding problems and see how your model responds!")
    print("Type 'quit' to exit.\n")
    
    while True:
        prompt = input("🤔 Enter a coding problem: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        print("🤖 Generating solution...")
        solution = generate_solution(model, tokenizer, prompt)
        
        print(f"📝 Solution:\n{solution}")
        
        # Optional: Get AI feedback
        print("\n🧠 Getting AI feedback...")
        # For interactive mode, we'll just evaluate code quality since we don't have test cases
        reward = reward_model._evaluate_code_quality(solution) + reward_model._ai_evaluate_solution(prompt, solution)
        reward = min(1.0, reward)  # Cap at 1.0
        
        print(f"🎯 AI Quality Score: {reward:.3f}")
        
        if reward > 0.6:
            print("✅ Good solution!")
        else:
            print("⚠️ Could be improved")
        
        print("-" * 50)
    
    print("👋 Thanks for testing!")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RLAIF trained model")
    parser.add_argument("--model_path", default="./trained_model", help="Path to trained model")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test(args.model_path)
    else:
        evaluate_model(args.model_path)

if __name__ == "__main__":
    main()