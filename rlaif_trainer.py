"""
Core RLAIF Trainer - Reinforcement Learning from AI Feedback
This demonstrates the key concepts of RLAIF for code generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from reward_model import AIRewardModel, create_reward_function

class RLAIFTrainer:
    """
    Simple RLAIF trainer that demonstrates the core concepts:
    1. Policy model (the model we're training)
    2. AI reward model (replaces human feedback)
    3. RL training loop using PPO
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """Initialize RLAIF trainer."""
        print("🤖 Initializing RLAIF Trainer...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize policy model (the model we're training)
        print(f"Loading policy model: {model_name}")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Initialize reference model (for KL penalty)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Initialize AI reward model (this is the key RLAIF component!)
        self.reward_model = AIRewardModel()
        
        # PPO configuration
        self.ppo_config = PPOConfig(
            output_dir="./ppo_output",
            learning_rate=1e-5,
            batch_size=2,
            mini_batch_size=1,
            bf16=False,
            fp16=False,
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            args=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
        )
        
        print("✅ RLAIF Trainer initialized successfully!")
    
    def train(self, train_dataset: Dataset, num_episodes: int = 10):
        """
        Main RLAIF training loop.
        
        Args:
            train_dataset: Dataset with coding problems
            num_episodes: Number of training episodes
        """
        print(f"🚀 Starting RLAIF training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # Sample batch from dataset
            batch_size = min(self.ppo_config.batch_size, len(train_dataset))
            batch_indices = np.random.choice(len(train_dataset), batch_size, replace=False)
            batch = train_dataset.select(batch_indices)
            
            # Extract queries (coding problems)
            queries = [item['query'] for item in batch]
            query_tensors = [self.tokenizer.encode(query, return_tensors="pt")[0] for query in queries]
            
            # Generate responses using current policy
            print("🤖 Generating code solutions...")
            response_tensors = self._generate_responses(query_tensors)
            
            # Decode responses
            responses = [self.tokenizer.decode(response, skip_special_tokens=True) for response in response_tensors]
            
            # Get AI rewards (this is the RLAIF magic!)
            print("🧠 Getting AI feedback...")
            rewards = self._get_ai_rewards(batch, responses)
            
            # Convert rewards to tensors
            reward_tensors = [torch.tensor(reward, dtype=torch.float) for reward in rewards]
            
            # Run PPO step
            print("🔄 Running PPO update...")
            stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
            
            # Log progress
            avg_reward = np.mean(rewards)
            print(f"Average reward: {avg_reward:.3f}")
            
            if stats:
                print(f"Policy loss: {stats.get('ppo/loss/policy', 0):.4f}")
                print(f"Value loss: {stats.get('ppo/loss/value', 0):.4f}")
                print(f"KL divergence: {stats.get('ppo/val/mean_kl', 0):.4f}")
        
        print("\n✅ RLAIF training completed!")
    
    def _generate_responses(self, query_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate code solutions using the current policy."""
        response_tensors = []
        
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 100,  # Limit response length
        }
        
        for query_tensor in query_tensors:
            query_tensor = query_tensor.unsqueeze(0)  # Add batch dimension
            
            # Generate response
            response_tensor = self.ppo_trainer.generate(
                query_tensor, 
                return_prompt=False,
                **generation_kwargs
            )
            
            response_tensors.append(response_tensor[0])
        
        return response_tensors
    
    def _get_ai_rewards(self, batch: Dataset, responses: List[str]) -> List[float]:
        """Get rewards from AI reward model (core RLAIF component)."""
        rewards = []
        
        for i, response in enumerate(responses):
            item = batch[i]
            
            # Use AI reward model to evaluate the response
            reward = self.reward_model.evaluate_code_solution(
                prompt=item['query'],
                solution=response,
                test_input=item['test_input'],
                expected_output=item['expected_output']
            )
            
            rewards.append(reward)
            
            # Debug: show what the model generated
            if i == 0:  # Show first example
                print(f"Generated: {response[:100]}...")
                print(f"Reward: {reward:.3f}")
        
        return rewards
    
    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the trained model."""
        print("🔍 Evaluating model...")
        
        total_reward = 0.0
        successful_generations = 0
        
        for item in test_dataset:
            try:
                # Generate solution
                query = item['query']
                query_tensor = self.tokenizer.encode(query, return_tensors="pt")
                
                with torch.no_grad():
                    response_tensor = self.model.generate(
                        query_tensor,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(response_tensor[0][query_tensor.shape[1]:], skip_special_tokens=True)
                
                # Get AI reward
                reward = self.reward_model.evaluate_code_solution(
                    prompt=query,
                    solution=response,
                    test_input=item['test_input'],
                    expected_output=item['expected_output']
                )
                
                total_reward += reward
                if reward > 0.5:  # Consider > 0.5 as successful
                    successful_generations += 1
                
                print(f"Problem: {query[:50]}...")
                print(f"Generated: {response[:80]}...")
                print(f"Reward: {reward:.3f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error evaluating: {e}")
        
        avg_reward = total_reward / len(test_dataset)
        success_rate = successful_generations / len(test_dataset)
        
        results = {
            "average_reward": avg_reward,
            "success_rate": success_rate,
            "total_problems": len(test_dataset)
        }
        
        print(f"📊 Evaluation Results:")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Success Rate: {success_rate:.3f}")
        
        return results
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        print(f"💾 Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("✅ Model saved successfully!")
    
    def generate_sample(self, prompt: str) -> str:
        """Generate a single code solution for testing."""
        query_tensor = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            response_tensor = self.model.generate(
                query_tensor,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(response_tensor[0][query_tensor.shape[1]:], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    # Quick test of the trainer
    from simple_dataset import create_dataset
    
    print("Testing RLAIF Trainer...")
    
    # Create dataset
    train_dataset, test_dataset = create_dataset()
    
    # Initialize trainer
    trainer = RLAIFTrainer()
    
    # Test generation before training
    print("\n🔍 Before training:")
    sample_prompt = "Write a Python function that adds two numbers."
    response = trainer.generate_sample(sample_prompt)
    print(f"Generated: {response}")
    
    # Run a few training episodes
    trainer.train(train_dataset, num_episodes=3)
    
    # Test after training
    print("\n🔍 After training:")
    response = trainer.generate_sample(sample_prompt)
    print(f"Generated: {response}")
    
    # Evaluate
    trainer.evaluate(test_dataset)