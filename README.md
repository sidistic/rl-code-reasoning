# RLAIF for Code Generation

A beginner-friendly implementation of RLAIF (Reinforcement Learning from AI Feedback) for code generation in just 5 core scripts!

## 📁 Project Structure

```
├── scrape_leetcode.py   # 1. Scrape coding problems
├── train.py            # 2. Train model (local or SageMaker)
├── evaluate.py         # 3. Evaluate trained model
├── reward_model.py     # 4. Modular reward system
├── rlaif_trainer.py    # 5. Core RLAIF implementation
└── requirements.txt
```

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets accelerate tqdm
# For SageMaker: pip install sagemaker boto3
# For LoRA: pip install peft
```

### 2. Get Training Data
```bash
# Scrape Leetcode problems (or use mock data)
python scrape_leetcode.py
```

### 3. Train Your Model

**Local Training (Quick Test):**
```bash
# Small model, few episodes
python train.py --model microsoft/DialoGPT-small --episodes 5

# Better model, more training
python train.py --model Salesforce/codegen-350M-mono --episodes 20
```

**SageMaker Training (With GPU):**
```bash
python train.py --mode sagemaker --model Salesforce/codegen-350M-mono --episodes 30
```

### 4. Evaluate Results
```bash
# Test on problems
python evaluate.py

# Interactive mode
python evaluate.py --interactive
```

## 🧠 Understanding RLAIF

**Traditional RLHF:** Human rates outputs → Expensive & slow  
**RLAIF Innovation:** AI rates outputs → Fast & scalable

### How It Works:
1. **Generate:** Model creates code solutions
2. **Evaluate:** AI gives rewards (execution, correctness, style)
3. **Learn:** Model updates to generate better code
4. **Repeat:** Iterate until performance improves

## 🔧 Customization

### Different Reward Models
Edit `reward_model.py`:
```python
# Execution-focused
model = CodeExecutionReward(weights={
    "correctness": 0.7,
    "execution": 0.2,
    "style": 0.1
})

# AI feedback
model = AIFeedbackReward()

# Hybrid approach
model = create_reward_model("hybrid")
```

### Add Your Own Problems
Edit `scrape_leetcode.py` or create JSON:
```json
{
  "prompt": "Write a function to find max in array",
  "solution": "def find_max(arr):\n    return max(arr)",
  "test_cases": [
    {"input": "find_max([1,5,3])", "output": "5"}
  ]
}
```

### Scale Up
```bash
# Larger models
python train.py --model codellama/CodeLlama-7b-Python-hf --use_lora

# More episodes
python train.py --episodes 100

# Bigger batches (edit rlaif_trainer.py)
batch_size = 8
```

## 📊 Expected Results

**Before Training:**
```
Prompt: Write a function to add two numbers
Output: Hello world! I like coding...
Reward: 0.1
```

**After Training:**
```
Prompt: Write a function to add two numbers  
Output: def add(a, b):
    return a + b
Reward: 0.95
```

## 💡 Key Concepts

- **Policy Model:** The model learning to generate code
- **Reward Model:** AI that evaluates code quality
- **PPO:** Algorithm that updates model carefully
- **Value Head:** Predicts expected rewards
- **Advantages:** How much better/worse than expected

## 🐛 Troubleshooting

**Out of Memory:**
- Use smaller model: `--model microsoft/DialoGPT-small`
- Enable LoRA: `--use_lora`
- Reduce batch size in `rlaif_trainer.py`

**No Improvement:**
- Increase episodes: `--episodes 50`
- Check reward model is working
- Verify dataset quality

**Slow Training:**
- Use GPU (CUDA)
- Try SageMaker with better instance
- Start with fewer problems

## 🚀 Next Steps

1. **Experiment with Rewards:** Modify `reward_model.py` weights
2. **Add Problems:** Expand dataset with real Leetcode problems  
3. **Try Larger Models:** CodeLlama, StarCoder, etc.
4. **Advanced RL:** Implement DPO, add KL penalties
5. **Production:** Add logging, checkpointing, distributed training

## 📚 Learn More

- [Original RLHF Paper](https://arxiv.org/abs/2203.02155)
- [Constitutional AI (RLAIF)](https://arxiv.org/abs/2212.08073)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

---

**Happy Learning!** 🎉 You now have a working RLAIF system that's simple to understand and easy to extend!

## 📚 Key Papers

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Original RLHF)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (RLAIF concept)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Alternative to PPO)

## 🤝 Contributing

This is a learning-focused implementation. Feel free to:
- Add more sophisticated reward functions
- Implement additional RL algorithms  
- Create better evaluation metrics
- Add more comprehensive datasets

---

**🎉 Happy Learning!** You now have a working RLAIF system in ~500 lines of code!