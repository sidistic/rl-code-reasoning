"""
RLAIF Trainer for Code Generation using Code-specific models
Optimized for GPU training on SageMaker
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model

from reward_model import AIRewardModel

class RLAIFCodeTrainer:
    """
    RLAIF trainer optimized for code generation models.
    Uses LoRA for efficient training on smaller GPUs.
    """
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Python-hf", use_lora: bool = True):
        """Initialize RLAIF trainer with code-specific model."""
        print(f"🤖 Initializing RLAIF Code Trainer with {model_name}...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        if "7b" in model_name.lower():
            # Use 8-bit quantization for 7B models
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            # Smaller models can run in fp16
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Apply LoRA for efficient training
        if use_lora and "7b" in model_name.lower():
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            print("✅ LoRA applied for efficient training")
        
        # Initialize reference model (frozen)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            load_in_8bit=True if "7b" in model_name.lower() else False,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Initialize AI reward model (use smaller model)
        self.reward_model = AIRewardModel("Salesforce/codet5-small")
        
        # PPO configuration optimized for code generation
        self.ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=1e-5,
            batch_size=4,
            mini_batch_size=2,
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            log_with="tensorboard",
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        print("✅ RLAIF Code Trainer initialized successfully!")
    
    def train(self, train_dataset: Dataset, num_episodes: int = 10):
        """
        Main RLAIF training loop optimized for code generation.
        """
        print(f"🚀 Starting RLAIF training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # Sample batch
            batch_size = min(4, len(train_dataset))
            batch_indices = np.random.choice(len(train_dataset), batch_size, replace=False)
            batch = train_dataset.select(batch_indices)
            
            # Prepare queries
            queries = [item['query'] for item in batch]
            query_tensors = [
                self.tokenizer.encode(query, return_tensors="pt", max_length=256, truncation=True).squeeze()
                for query in queries
            ]
            
            # Generate responses
            print("🤖 Generating code solutions...")
            response_tensors = []
            
            for query_tensor in query_tensors:
                gen_kwargs = {
                    "min_new_tokens": 10,
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.95,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                
                response = self.ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    return_prompt=False,
                    **gen_kwargs
                )
                response_tensors.append(response.squeeze())
            
            # Decode responses
            responses = [
                self.tokenizer.decode(r, skip_special_tokens=True) 
                for r in response_tensors
            ]
            
            # Get AI rewards
            print("🧠 Getting AI feedback...")
            rewards = []
            
            for i, (response, item) in enumerate(zip(responses, batch)):
                reward = self.reward_model.evaluate_code_solution(
                    prompt=item['query'],
                    solution=response,
                    test_input=item['test_input'],
                    expected_output=item['expected_output']
                )
                rewards.append(torch.tensor(reward))
                
                if i == 0:  # Show first example
                    print(f"Query: {item['query'][:60]}...")
                    print(f"Generated: {response[:80]}...")
                    print(f"Reward: {reward:.3f}")
            
            # Run PPO step
            print("🔄 Running PPO update...")
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log stats
            avg_reward = torch.stack(rewards).mean().item()
            print(f"Average reward: {avg_reward:.3f}")
            
            if stats and "ppo/loss/policy" in stats:
                print(f"Policy loss: {stats['ppo/loss/policy']:.4f}")
                print(f"Value loss: {stats['ppo/loss/value']:.4f}")
        
        print("\n✅ RLAIF training completed!")
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        print(f"💾 Saving model to {output_dir}")
        
        # Save the PPO model
        self.ppo_trainer.save_pretrained(output_dir)
        
        print("✅ Model saved successfully!")
    
    def generate_sample(self, prompt: str) -> str:
        """Generate a code solution for testing."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response


if __name__ == "__main__":
    # Quick test
    from simple_dataset import create_dataset
    
    print("Testing RLAIF Code Trainer...")
    
    # Use small model for testing
    trainer = RLAIFCodeTrainer("Salesforce/codegen-350M-mono")
    
    # Create dataset
    train_dataset, _ = create_dataset()
    
    # Test generation
    prompt = "Write a Python function that adds two numbers."
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {trainer.generate_sample(prompt)}")
    
    # Run one episode
    trainer.train(train_dataset, num_episodes=1)