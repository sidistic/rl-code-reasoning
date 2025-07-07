"""
Simple RLAIF Training Script
Run this to start training your code generation model with AI feedback.
"""

import argparse
import torch
from simple_dataset import create_dataset, save_dataset
from rlaif_trainer import RLAIFTrainer

def main():
    parser = argparse.ArgumentParser(description="Train code generation model with RLAIF")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="Base model to train")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--output_dir", default="./trained_model", help="Where to save the model")
    parser.add_argument("--create_data", action="store_true", help="Create new dataset")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    print("🧠 RLAIF Code Generation Training")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU (training will be slow)")
    print()
    
    # Create or load dataset
    if args.create_data:
        print("📊 Creating new dataset...")
        train_dataset, test_dataset = save_dataset()
    else:
        print("📊 Loading dataset...")
        train_dataset, test_dataset = create_dataset()
    
    # Initialize trainer
    print("🤖 Initializing RLAIF trainer...")
    trainer = RLAIFTrainer(model_name=args.model)
    
    if not args.eval_only:
        # Show example before training
        print("\n🔍 Example generation BEFORE training:")
        sample_prompt = "Write a Python function that adds two numbers."
        before_response = trainer.generate_sample(sample_prompt)
        print(f"Prompt: {sample_prompt}")
        print(f"Generated: {before_response.strip()}")
        
        # Train the model
        print(f"\n🚀 Starting RLAIF training...")
        trainer.train(train_dataset, num_episodes=args.episodes)
        
        # Show example after training
        print("\n🔍 Example generation AFTER training:")
        after_response = trainer.generate_sample(sample_prompt)
        print(f"Prompt: {sample_prompt}")
        print(f"Generated: {after_response.strip()}")
        
        # Save the model
        trainer.save_model(args.output_dir)
    
    # Evaluate the model
    print("\n📊 Evaluating model performance...")
    results = trainer.evaluate(test_dataset)
    
    print("\n🎉 Training completed!")
    print(f"Final average reward: {results['average_reward']:.3f}")
    print(f"Success rate: {results['success_rate']:.3f}")
    
    if not args.eval_only:
        print(f"Model saved to: {args.output_dir}")

def demo():
    """Run a quick demo to show how RLAIF works."""
    print("🎮 RLAIF Demo - See How AI Feedback Works")
    print("=" * 50)
    
    # Create small dataset
    train_dataset, test_dataset = create_dataset()
    
    # Show the concept
    from reward_model import AIRewardModel
    
    print("1️⃣ Creating AI Reward Model...")
    reward_model = AIRewardModel()
    
    print("\n2️⃣ Testing AI Feedback on different solutions:")
    
    prompt = "Write a Python function that adds two numbers."
    test_input = "add_numbers(3, 5)"
    expected_output = "8"
    
    # Good solution
    good_solution = """def add_numbers(a, b):
    return a + b"""
    
    # Bad solution
    bad_solution = """print("hello world")"""
    
    # Test both
    good_reward = reward_model.evaluate_code_solution(prompt, good_solution, test_input, expected_output)
    bad_reward = reward_model.evaluate_code_solution(prompt, bad_solution, test_input, expected_output)
    
    print(f"\n✅ Good solution reward: {good_reward:.3f}")
    print(f"❌ Bad solution reward: {bad_reward:.3f}")
    
    print(f"\n3️⃣ This is how RLAIF works:")
    print("- AI evaluates solutions instead of humans")
    print("- Higher rewards for better code")
    print("- Model learns to maximize AI rewards")
    print("- Result: Better code generation!")
    
    print(f"\n🚀 Now run: python train.py --episodes 10")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo()
    else:
        # Arguments provided, run training
        main()