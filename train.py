"""
Local RLAIF Training Script for Code Generation
Test locally before running on SageMaker.
"""

import argparse
import torch
from simple_dataset import create_dataset
from rlaif_trainer import RLAIFCodeTrainer

def main():
    parser = argparse.ArgumentParser(description="Train code generation model with RLAIF")
    
    # Model selection
    parser.add_argument("--model", default="Salesforce/codegen-350M-mono", 
                       choices=[
                           "microsoft/DialoGPT-small",  # For CPU testing
                           "Salesforce/codegen-350M-mono",
                           "WizardLM/WizardCoder-1B-V1.0",
                           "Salesforce/codegen-2B-mono",
                           "codellama/CodeLlama-7b-Python-hf"
                       ],
                       help="Model to train")
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--output_dir", default="./trained_model", help="Where to save model")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for 7B models")
    
    args = parser.parse_args()
    
    print("🧠 RLAIF Code Generation Training (Local)")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU - training will be slow!")
        if "7b" in args.model.lower():
            print("❌ Cannot run 7B models without GPU!")
            return
    
    # Create dataset
    print("\n📊 Creating dataset...")
    train_dataset, test_dataset = create_dataset()
    
    # Initialize trainer
    print("\n🤖 Initializing RLAIF trainer...")
    try:
        trainer = RLAIFCodeTrainer(
            model_name=args.model,
            use_lora=args.use_lora or "7b" in args.model.lower()
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Try a smaller model like 'Salesforce/codegen-350M-mono'")
        return
    
    # Show initial performance
    print("\n🔍 Testing BEFORE training:")
    test_prompts = [
        "Write a Python function that adds two numbers.",
        "Write a Python function that checks if a number is even."
    ]
    
    for prompt in test_prompts[:1]:
        response = trainer.generate_sample(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {response.strip()[:100]}...")
        print()
    
    # Train
    print(f"🚀 Starting RLAIF training for {args.episodes} episodes...")
    trainer.train(train_dataset, num_episodes=args.episodes)
    
    # Show final performance
    print("\n🔍 Testing AFTER training:")
    for prompt in test_prompts[:1]:
        response = trainer.generate_sample(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {response.strip()[:100]}...")
        print()
    
    # Save model
    trainer.save_model(args.output_dir)
    print(f"\n✅ Model saved to {args.output_dir}")
    print("\n🎉 Training completed!")

def demo():
    """Quick demo showing RLAIF concept."""
    print("🎮 RLAIF Demo - Code Generation with AI Feedback")
    print("=" * 60)
    
    from reward_model import AIRewardModel
    
    # Create reward model
    print("1️⃣ Loading AI Reward Model...")
    reward_model = AIRewardModel("Salesforce/codet5-small")
    
    print("\n2️⃣ Testing AI feedback on code solutions:")
    
    prompt = "Write a Python function that adds two numbers."
    
    # Test different solutions
    solutions = [
        ("✅ Good", "def add_numbers(a, b):\n    return a + b"),
        ("❌ Bad", "print('hello world')"),
        ("🔄 Okay", "def add(x, y):\n    result = x + y")
    ]
    
    for label, solution in solutions:
        reward = reward_model.evaluate_code_solution(
            prompt, solution, "add_numbers(3, 5)", "8"
        )
        print(f"\n{label} solution:")
        print(f"Code: {solution}")
        print(f"AI Reward: {reward:.3f}")
    
    print("\n3️⃣ How RLAIF trains:")
    print("• Model generates code")
    print("• AI evaluates quality (no humans!)")
    print("• Model learns from AI rewards")
    print("• Repeat → Better code!")
    
    print("\n🚀 Ready to train? Try these commands:")
    print("\nLocal (small model):")
    print("  python train.py --model Salesforce/codegen-350M-mono --episodes 10")
    print("\nSageMaker (with GPU):")
    print("  python launch_sagemaker_training.py --model Salesforce/codegen-350M-mono --episodes 30")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        demo()
    else:
        main()