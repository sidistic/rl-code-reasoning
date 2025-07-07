"""
RLAIF Training Script
Supports both local and AWS SageMaker training
"""

import argparse
import json
import os
import torch
from datasets import Dataset

def load_dataset(file_path="leetcode_problems.json"):
    """Load training data"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        # Fallback to simple problems
        data = [
            {
                "prompt": "Write a Python function that adds two numbers.",
                "solution": "def add(a, b):\n    return a + b",
                "test_cases": [{"input": "add(2, 3)", "output": "5"}]
            },
            {
                "prompt": "Write a Python function that checks if a number is even.",
                "solution": "def is_even(n):\n    return n % 2 == 0",
                "test_cases": [{"input": "is_even(4)", "output": "True"}]
            }
        ]
    
    # Format for training
    formatted = []
    for item in data:
        formatted.append({
            "query": item["prompt"],
            "test_input": item["test_cases"][0]["input"] if item["test_cases"] else "",
            "expected_output": item["test_cases"][0]["output"] if item["test_cases"] else "",
        })
    
    return Dataset.from_list(formatted)

def train_local(args):
    """Train model locally"""
    from rlaif_trainer import RLAIFTrainer
    
    print(f"🖥️ Training locally with {args.model}")
    print(f"Episodes: {args.episodes}")
    
    # Load dataset
    dataset = load_dataset(args.data)
    
    # Initialize trainer
    trainer = RLAIFTrainer(
        model_name=args.model,
        use_lora=args.use_lora
    )
    
    # Train
    trainer.train(dataset, num_episodes=args.episodes)
    
    # Save
    trainer.save_model(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

def train_sagemaker(args):
    """Train on AWS SageMaker"""
    import sagemaker
    from sagemaker.pytorch import PyTorch
    
    print(f"☁️ Training on SageMaker with {args.model}")
    
    # SageMaker setup
    sagemaker_session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()
    
    # Create training script for SageMaker
    train_code = """
import os
import sys
sys.path.append('/opt/ml/code')
from train import train_local
import argparse

# Parse SageMaker hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=os.environ.get('SM_HP_MODEL_NAME'))
parser.add_argument('--episodes', type=int, default=int(os.environ.get('SM_HP_EPISODES', 10)))
parser.add_argument('--use_lora', type=bool, default=os.environ.get('SM_HP_USE_LORA', 'True') == 'True')
parser.add_argument('--output_dir', type=str, default='/opt/ml/model')
parser.add_argument('--data', type=str, default='/opt/ml/input/data/training/leetcode_problems.json')
args = parser.parse_args()

# Train
train_local(args)
"""
    
    # Save training script
    os.makedirs("sagemaker_code", exist_ok=True)
    with open("sagemaker_code/train_entry.py", "w") as f:
        f.write(train_code)
    
    # Copy necessary files
    import shutil
    for file in ["rlaif_trainer.py", "reward_model.py", "train.py"]:
        if os.path.exists(file):
            shutil.copy(file, f"sagemaker_code/{file}")
    
    # Configure PyTorch estimator
    estimator = PyTorch(
        entry_point="train_entry.py",
        source_dir="sagemaker_code",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.0",
        py_version="py310",
        hyperparameters={
            "model_name": args.model,
            "episodes": args.episodes,
            "use_lora": args.use_lora
        },
        output_path=f"s3://{sagemaker_session.default_bucket()}/rlaif-output"
    )
    
    # Upload data to S3
    s3_data = sagemaker_session.upload_data(
        path=args.data,
        key_prefix="rlaif-data"
    )
    
    # Start training
    print("🚀 Starting SageMaker training job...")
    estimator.fit({"training": s3_data}, wait=args.wait)
    
    if args.wait:
        print(f"✅ Training completed!")
        print(f"Model location: {estimator.model_data}")

def main():
    parser = argparse.ArgumentParser(description="RLAIF Training")
    
    # Training mode
    parser.add_argument("--mode", choices=["local", "sagemaker"], default="local",
                       help="Training mode: local or sagemaker")
    
    # Model configuration
    parser.add_argument("--model", default="Salesforce/codegen-350M-mono",
                       help="Model to train")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of training episodes")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for efficient training")
    
    # Data and output
    parser.add_argument("--data", default="leetcode_problems.json",
                       help="Training data file")
    parser.add_argument("--output_dir", default="./trained_model",
                       help="Output directory for trained model")
    
    # SageMaker specific
    parser.add_argument("--instance_type", default="ml.g4dn.xlarge",
                       help="SageMaker instance type")
    parser.add_argument("--role", help="SageMaker IAM role")
    parser.add_argument("--wait", action="store_true",
                       help="Wait for SageMaker job to complete")
    
    args = parser.parse_args()
    
    # Show configuration
    print("🧠 RLAIF Code Generation Training")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Data: {args.data}")
    
    # Check GPU for local training
    if args.mode == "local" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Train
    if args.mode == "local":
        train_local(args)
    else:
        train_sagemaker(args)

if __name__ == "__main__":
    main()