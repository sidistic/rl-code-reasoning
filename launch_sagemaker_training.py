"""
Launch RLAIF training on AWS SageMaker
Run this script from your local machine to start training on SageMaker.
"""

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from datetime import datetime
import argparse

def launch_training(
    model_name="Salesforce/codegen-350M-mono",
    instance_type="ml.g4dn.xlarge",
    episodes=20
):
    """Launch RLAIF training job on SageMaker."""
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()  # Or specify your role ARN
    
    # S3 bucket for outputs
    bucket = sagemaker_session.default_bucket()
    prefix = f"rlaif-code-generation/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    print(f"🚀 Launching RLAIF training on SageMaker")
    print(f"Model: {model_name}")
    print(f"Instance: {instance_type}")
    print(f"Episodes: {episodes}")
    print(f"S3 Output: s3://{bucket}/{prefix}")
    
    # Define PyTorch estimator
    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=".",  # Current directory with all Python files
        role=role,
        instance_type=instance_type,
        instance_count=1,
        framework_version="2.0",
        py_version="py310",
        hyperparameters={
            "model_name": model_name,
            "episodes": episodes,
            "use_lora": True
        },
        output_path=f"s3://{bucket}/{prefix}/output",
        base_job_name="rlaif-code-training",
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"  # Help with memory
        }
    )
    
    # Start training
    print("\n📊 Starting training job...")
    estimator.fit(wait=False)  # Set wait=True to wait for completion
    
    print(f"\n✅ Training job submitted!")
    print(f"Job name: {estimator.latest_training_job.name}")
    print(f"Monitor in SageMaker console or run:")
    print(f"  estimator.logs()")
    
    return estimator

def main():
    parser = argparse.ArgumentParser(description="Launch RLAIF training on SageMaker")
    
    # Model selection
    parser.add_argument("--model", type=str, default="Salesforce/codegen-350M-mono",
                       choices=[
                           "Salesforce/codegen-350M-mono",
                           "Salesforce/codegen-2B-mono",
                           "codellama/CodeLlama-7b-Python-hf",
                           "bigcode/starcoder",
                           "WizardLM/WizardCoder-1B-V1.0"
                       ],
                       help="Model to train")
    
    # Instance selection based on model size
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge",
                       help="SageMaker instance type")
    
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of training episodes")
    
    args = parser.parse_args()
    
    # Recommend instance based on model
    if "7b" in args.model.lower() or "starcoder" in args.model.lower():
        if args.instance_type == "ml.g4dn.xlarge":
            print("⚠️ For 7B models, consider using ml.g4dn.2xlarge or larger")
            response = input("Continue with ml.g4dn.xlarge? (y/n): ")
            if response.lower() != 'y':
                args.instance_type = "ml.g4dn.2xlarge"
                print(f"Using {args.instance_type}")
    
    # Launch training
    estimator = launch_training(
        model_name=args.model,
        instance_type=args.instance_type,
        episodes=args.episodes
    )
    
    # Optional: wait and download model
    response = input("\nWait for training to complete? (y/n): ")
    if response.lower() == 'y':
        print("⏳ Waiting for training to complete...")
        estimator.logs()  # This will stream logs and wait
        
        # Download model
        print("\n📥 Downloading trained model...")
        model_data = estimator.model_data
        print(f"Model location: {model_data}")
        
        # You can download using AWS CLI:
        print("\nTo download the model, run:")
        print(f"aws s3 cp {model_data} ./trained_model.tar.gz")
        print("tar -xzf trained_model.tar.gz")

if __name__ == "__main__":
    main()