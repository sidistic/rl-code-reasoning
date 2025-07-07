# 🤖 Simple RLAIF for Code Generation

A minimal implementation of **RLAIF (Reinforcement Learning from AI Feedback)** for learning code generation. This project demonstrates the core concepts of RLAIF in under 500 lines of code.

## 🎯 What is RLAIF?

**RLAIF** replaces human feedback with AI feedback in reinforcement learning:
- **Traditional RLHF**: Human evaluators rate model outputs → expensive and slow
- **RLAIF**: AI model evaluates outputs → scalable and fast

## 📁 Project Structure (6 files only!)

```
├── requirements.txt        # Dependencies
├── simple_dataset.py       # Creates coding problems  
├── reward_model.py         # AI reward model (core RLAIF!)
├── rlaif_trainer.py        # Main RLAIF training logic
├── train.py               # Training script
├── evaluate.py           # Evaluation script
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. See RLAIF Demo
```bash
python train.py
```
This shows how AI feedback works on good vs bad code.

### 3. Train Your Model
```bash
python train.py --episodes 20
```

### 4. Evaluate Results
```bash
python evaluate.py
```

### 5. Interactive Testing
```bash
python evaluate.py --interactive
```

## 🧠 How RLAIF Works

1. **Policy Model**: The model we're training (starts with DialoGPT)
2. **AI Reward Model**: Evaluates code quality (replaces humans!)
3. **Training Loop**: 
   - Generate code solutions
   - Get AI rewards 
   - Update policy with PPO
   - Repeat!

## 💡 Key Components

### AI Reward Model (`reward_model.py`)
```python
# This is the magic of RLAIF!
reward = ai_model.evaluate_code_solution(
    prompt="Write a function to add numbers",
    solution=generated_code,
    test_input="add(3,5)", 
    expected_output="8"
)
# Higher reward = better code
```

### RLAIF Training (`rlaif_trainer.py`)
```python
# Core RLAIF training loop
for episode in range(num_episodes):
    # 1. Generate solutions with current policy
    responses = model.generate(prompts)
    
    # 2. Get AI feedback (not human!)
    rewards = ai_reward_model.evaluate(responses)
    
    # 3. Update policy to maximize AI rewards
    ppo_trainer.step(prompts, responses, rewards)
```

## 📊 Example Training Output

```
🤖 Episode 1/20
🤖 Generating code solutions...
🧠 Getting AI feedback...
Generated: def add_numbers print hello
Reward: 0.245

🔄 Running PPO update...
Average reward: 0.245

...

🤖 Episode 20/20  
Generated: def add_numbers(a, b):
    return a + b
Reward: 0.847

✅ RLAIF training completed!
```

## 🎛️ Configuration Options

### Basic Training
```bash
python train.py --episodes 10        # Quick training
python train.py --episodes 50        # Better results
```

### Different Models
```bash
python train.py --model microsoft/DialoGPT-medium
python train.py --model gpt2
```

### Custom Output
```bash
python train.py --output_dir ./my_model
```

## 📈 Understanding Results

**Reward Scores:**
- `0.0-0.3`: Poor code (syntax errors, wrong logic)
- `0.3-0.6`: Okay code (works but inefficient)  
- `0.6-1.0`: Good code (correct, clean, efficient)

**Success Rate:**
- `> 70%`: Excellent training
- `40-70%`: Good progress
- `< 40%`: Needs more episodes

## 🔬 What You'll Learn

1. **RLAIF Basics**: How AI feedback replaces human evaluation
2. **PPO Training**: Reinforcement learning for language models
3. **Reward Design**: How to create effective AI reward functions
4. **Code Evaluation**: Automated assessment of generated code

## 🛠️ Customization Ideas

### Add New Problems
Edit `simple_dataset.py`:
```python
problems = [
    {
        "prompt": "Write a function to sort a list",
        "test_input": "sort_list([3,1,4,1,5])",
        "expected_output": "[1,1,3,4,5]",
        "difficulty": "medium"
    }
]
```

### Improve Reward Model
Edit `reward_model.py`:
```python
def _evaluate_code_quality(self, solution):
    score = 0.0
    
    # Add your own criteria
    if "docstring" in solution:
        score += 0.1
    if "error handling" in solution:
        score += 0.1
        
    return score
```

### Use Better Base Models
```python
# In train.py, try:
trainer = RLAIFTrainer("microsoft/CodeBERT-base")
trainer = RLAIFTrainer("Salesforce/codegen-350M-mono")
```

## 🐛 Troubleshooting

**GPU Memory Issues:**
```bash
# Use smaller model
python train.py --model microsoft/DialoGPT-small

# Reduce batch size (edit rlaif_trainer.py)
batch_size=1
```

**Slow Training:**
- Start with `--episodes 5` for testing
- Use GPU if available
- Consider smaller models for experimentation

**Poor Results:**
- Increase episodes: `--episodes 50`
- Check if rewards are increasing
- Verify test cases in dataset

## 🎯 Next Steps

Once you understand this simple RLAIF implementation:

1. **Scale Up**: Use larger models (CodeT5, StarCoder)
2. **Better Rewards**: Add execution testing, style checking  
3. **More Data**: Create larger, more diverse problem sets
4. **Advanced RL**: Try different algorithms (DPO, PPO variants)

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