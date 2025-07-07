# 🚀 Quick Start: RLAIF Code Generation on AWS SageMaker

Train your code generation model using RLAIF on AWS SageMaker with GPU support!

## 📋 Prerequisites

1. **AWS Account** with SageMaker access
2. **AWS CLI** configured with credentials
3. **SageMaker execution role** with S3 access

## 🎯 Model Options (Start Small!)

| Model | Size | GPU Memory | Instance Type | Training Speed |
|-------|------|------------|---------------|----------------|
| **Salesforce/codegen-350M-mono** | 350M | ~4GB | ml.g4dn.xlarge | Fast ⚡ |
| **WizardLM/WizardCoder-1B-V1.0** | 1B | ~6GB | ml.g4dn.xlarge | Good 👍 |
| **Salesforce/codegen-2B-mono** | 2B | ~10GB | ml.g4dn.xlarge | Moderate 🚀 |
| **codellama/CodeLlama-7b-Python-hf** | 7B | ~20GB | ml.g4dn.2xlarge | Slow 🐌 |

**💡 Recommendation: Start with codegen-350M for quick results!**

## 🏃 Quick Start (5 Minutes)

### 1. Install Requirements
```bash
pip install sagemaker boto3
```

### 2. Launch Training (Simplest Option)
```bash
# Train small model (fast, visible improvements)
python launch_sagemaker_training.py --model Salesforce/codegen-350M-mono --episodes 30

# Or train medium model (better results)
python launch_sagemaker_training.py --model WizardLM/WizardCoder-1B-V1.0 --episodes 20
```

### 3. Monitor Training
Go to AWS SageMaker console → Training jobs → View logs

## 📊 Expected Results

### Before Training:
```
Prompt: Write a Python function that adds two numbers.
Generated: Hello world! This is a function...
```

### After Training (350M model, 30 episodes):
```
Prompt: Write a Python function that adds two numbers.
Generated: def add_numbers(a, b):
    return a + b
```

## 💰 Cost Estimates

| Instance | Model Size | Cost/Hour | 30 Episodes Time | Total Cost |
|----------|------------|-----------|------------------|------------|
| ml.g4dn.xlarge | 350M-2B | $0.736 | ~30 mins | ~$0.40 |
| ml.g4dn.2xlarge | 7B | $1.12 | ~2 hours | ~$2.24 |

## 🛠️ Advanced Options

### Use Larger Model (Better Quality)
```bash
# CodeLlama 7B - needs bigger instance
python launch_sagemaker_training.py \
    --model codellama/CodeLlama-7b-Python-hf \
    --instance_type ml.g4dn.2xlarge \
    --episodes 15
```

### Local Testing First
```bash
# Test locally with tiny model
python train.py --model microsoft/DialoGPT-small --episodes 5
```

### Download Trained Model
After training completes:
```bash
# Get model location from SageMaker console or script output
aws s3 cp s3://your-bucket/path/model.tar.gz ./
tar -xzf model.tar.gz
```

## 🔧 Troubleshooting

### Out of Memory Error
- Use smaller model: `codegen-350M-mono` instead of `7B`
- Use larger instance: `ml.g4dn.2xlarge`
- Reduce batch size in `rlaif_trainer.py`

### Training Too Slow
- Use smaller model
- Reduce episodes: `--episodes 10`
- Use better instance type

### No Improvement
- Increase episodes: `--episodes 50`
- Check rewards are increasing in logs
- Ensure dataset has good examples

## 📈 What to Expect

**Good Training Progress:**
```
Episode 1/30
Average reward: 0.245 ❌

Episode 15/30
Average reward: 0.523 🔄

Episode 30/30
Average reward: 0.812 ✅
```

## 🎯 Next Steps

1. **Start Small**: Use `codegen-350M-mono` first
2. **Monitor Rewards**: Watch them increase during training
3. **Test Results**: Use `evaluate.py` after downloading model
4. **Scale Up**: Try larger models once working

## 💡 Pro Tips

- **Start with 350M model** - you'll see improvements in 20-30 episodes
- **Use spot instances** for 70% cost savings
- **Train overnight** with larger models
- **Save checkpoints** every 10 episodes for safety

---

**Ready to go?** Run:
```bash
python launch_sagemaker_training.py --model Salesforce/codegen-350M-mono --episodes 30
```

Your model will improve from random text to working Python functions! 🎉