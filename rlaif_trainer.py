"""
RLAIF Trainer - Educational Implementation
Clear implementation showing how RLAIF works for code generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
import logging

# Set up logging for educational purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLAIFTrainer:
    """
    RLAIF (Reinforcement Learning from AI Feedback) Trainer
    
    Key concepts:
    1. Policy Model: The model we're training to generate code
    2. Reward Model: AI that evaluates code quality (no humans!)
    3. PPO Algorithm: How we update the model based on rewards
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", use_lora: bool = False):
        """Initialize RLAIF components"""
        logger.info(f"🚀 Initializing RLAIF Trainer with {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"  # Important for generation
        
        # Load model (policy network)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # LoRA for efficient training (optional)
        if use_lora and model_name != "microsoft/DialoGPT-small":
            self._apply_lora()
        
        # Reference model (frozen copy for KL divergence)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.ref_model.eval()  # Always in eval mode
        
        # Load reward model
        from reward_model import create_reward_model
        self.reward_model = create_reward_model("hybrid")
        
        # PPO hyperparameters
        self.ppo_config = {
            "learning_rate": 1e-5,
            "kl_coef": 0.1,          # KL divergence coefficient
            "gamma": 0.99,           # Discount factor
            "gae_lambda": 0.95,      # GAE lambda
            "clip_ratio": 0.2,       # PPO clipping
            "value_loss_coef": 0.5,  # Value function coefficient
            "max_grad_norm": 0.5,    # Gradient clipping
        }
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ppo_config["learning_rate"])
        
        # Value head (estimates expected reward)
        hidden_size = self.model.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_head.parameters(), lr=self.ppo_config["learning_rate"])
        
        logger.info(f"✅ RLAIF Trainer ready on {self.device}")
    
    def _apply_lora(self):
        """Apply LoRA for efficient training"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("✅ LoRA applied for efficient training")
        except ImportError:
            logger.warning("⚠️ PEFT not installed, skipping LoRA")
    
    def generate_trajectories(self, prompts: List[str], max_length: int = 100) -> Dict:
        """
        Generate code solutions and collect trajectories
        
        Returns dict with:
        - responses: Generated code solutions
        - log_probs: Log probabilities of actions
        - values: Value estimates
        - masks: Attention masks
        """
        self.model.eval()
        trajectories = {
            "responses": [],
            "log_probs": [],
            "values": [],
            "rewards": []
        }
        
        with torch.no_grad():
            for prompt in prompts:
                # Encode prompt
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate with sampling
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Extract generated tokens
                generated_ids = outputs.sequences[0][inputs.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Get log probabilities
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    probs = torch.softmax(score[0], dim=-1)
                    token_id = generated_ids[i]
                    log_prob = torch.log(probs[token_id])
                    log_probs.append(log_prob.item())
                
                # Get value estimates
                last_hidden = self.model(inputs, output_hidden_states=True).hidden_states[-1]
                value = self.value_head(last_hidden[:, -1, :]).squeeze()
                
                trajectories["responses"].append(response)
                trajectories["log_probs"].append(log_probs)
                trajectories["values"].append(value.item())
        
        return trajectories
    
    def compute_rewards(self, prompts: List[str], responses: List[str], 
                       test_inputs: List[str], expected_outputs: List[str]) -> List[float]:
        """Get rewards from AI feedback"""
        rewards = self.reward_model.batch_compute(prompts, responses, test_inputs, expected_outputs)
        return rewards
    
    def compute_advantages(self, rewards: List[float], values: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        This tells us how much better/worse an action was compared to expected
        """
        advantages = []
        returns = []
        
        # Simple version: advantage = reward - value
        for reward, value in zip(rewards, values):
            returns.append(reward)  # No discounting for single-step
            advantages.append(reward - value)
        
        # Normalize advantages
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.tolist(), returns
    
    def ppo_step(self, prompts: List[str], old_log_probs: List[List[float]], 
                 advantages: List[float], returns: List[float]):
        """
        PPO update step - this is where learning happens!
        
        Key idea: Update policy to increase probability of good actions
        while staying close to old policy (for stability)
        """
        self.model.train()
        
        # Encode prompts
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        # Forward pass
        outputs = self.model(**encoded, output_hidden_states=True)
        
        # Compute new log probabilities
        # (simplified - in practice, need to match exact generation)
        logits = outputs.logits
        
        # Policy loss (PPO objective)
        policy_losses = []
        for i, advantage in enumerate(advantages):
            # Importance sampling ratio
            ratio = 1.0  # Simplified
            
            # PPO clipping
            clipped_ratio = torch.clamp(torch.tensor(ratio), 
                                      1 - self.ppo_config["clip_ratio"],
                                      1 + self.ppo_config["clip_ratio"])
            
            # Policy loss
            advantage_tensor = torch.tensor(advantage, dtype=torch.float32)
            policy_loss = -torch.min(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
            policy_losses.append(policy_loss)
        
        policy_loss = torch.stack(policy_losses).mean()
        
        # Value loss
        hidden_states = outputs.hidden_states[-1]
        predicted_values = self.value_head(hidden_states[:, -1, :]).squeeze()
        value_targets = torch.tensor(returns).to(self.device)
        value_loss = torch.nn.functional.mse_loss(predicted_values, value_targets)
        
        # Total loss
        total_loss = policy_loss + self.ppo_config["value_loss_coef"] * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config["max_grad_norm"])
        
        # Update
        self.optimizer.step()
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train(self, dataset, num_episodes: int = 10):
        """
        Main RLAIF training loop
        
        For each episode:
        1. Generate code solutions (trajectories)
        2. Get AI feedback (rewards)
        3. Compute advantages
        4. Update policy with PPO
        """
        logger.info(f"🏃 Starting RLAIF training for {num_episodes} episodes")
        
        # Training history
        history = {
            "rewards": [],
            "losses": []
        }
        
        for episode in range(num_episodes):
            logger.info(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # Sample batch from dataset
            batch_size = min(4, len(dataset))
            indices = np.random.choice(len(dataset), batch_size, replace=False)
            batch = dataset.select(indices)
            
            # Extract batch data
            prompts = [item['query'] for item in batch]
            test_inputs = [item['test_input'] for item in batch]
            expected_outputs = [item['expected_output'] for item in batch]
            
            # 1. Generate trajectories
            logger.info("🤖 Generating code solutions...")
            trajectories = self.generate_trajectories(prompts)
            
            # 2. Get rewards from AI
            logger.info("🧠 Getting AI feedback...")
            rewards = self.compute_rewards(prompts, trajectories["responses"], 
                                         test_inputs, expected_outputs)
            
            # Log example
            logger.info(f"Example - Prompt: {prompts[0][:50]}...")
            logger.info(f"Generated: {trajectories['responses'][0][:80]}...")
            logger.info(f"Reward: {rewards[0]:.3f}")
            
            # 3. Compute advantages
            advantages, returns = self.compute_advantages(rewards, trajectories["values"])
            
            # 4. PPO update
            logger.info("🔄 Running PPO update...")
            losses = self.ppo_step(prompts, trajectories["log_probs"], advantages, returns)
            
            # Track progress
            avg_reward = np.mean(rewards)
            history["rewards"].append(avg_reward)
            history["losses"].append(losses["total_loss"])
            
            logger.info(f"Average reward: {avg_reward:.3f}")
            logger.info(f"Loss: {losses['total_loss']:.4f}")
            
            # Early stopping if doing well
            if avg_reward > 0.9:
                logger.info("🎉 High performance achieved!")
                break
        
        logger.info("\n✅ RLAIF training completed!")
        return history
    
    def save_model(self, output_dir: str):
        """Save trained model"""
        logger.info(f"💾 Saving model to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save value head
        torch.save(self.value_head.state_dict(), f"{output_dir}/value_head.pt")
        
        logger.info("✅ Model saved!")

# Educational example showing RLAIF concept
def demonstrate_rlaif_concept():
    """Simple demonstration of RLAIF concepts"""
    print("\n🎓 RLAIF Concept Demonstration")
    print("=" * 50)
    
    print("\n1️⃣ Traditional RLHF:")
    print("   Model → Generated Code → Human Rates → Update Model")
    print("   Problem: Expensive, slow, doesn't scale")
    
    print("\n2️⃣ RLAIF Innovation:")
    print("   Model → Generated Code → AI Rates → Update Model")
    print("   Benefits: Fast, scalable, consistent")
    
    print("\n3️⃣ How it works:")
    print("   a) Model generates multiple solutions")
    print("   b) AI evaluates each solution (execution, style, correctness)")
    print("   c) Good solutions get high rewards")
    print("   d) Model learns to generate more like the good ones")
    
    print("\n4️⃣ PPO Algorithm:")
    print("   - Proximal Policy Optimization")
    print("   - Updates model carefully (not too much at once)")
    print("   - Prevents catastrophic forgetting")
    
    print("\n5️⃣ Key Components:")
    print("   - Policy Model: Generates code")
    print("   - Value Model: Predicts how good a solution will be")
    print("   - Reward Model: AI that evaluates code")
    print("   - PPO Optimizer: Updates model based on rewards")

if __name__ == "__main__":
    # Show concept
    demonstrate_rlaif_concept()
    
    # Quick test
    print("\n\n🧪 Quick Test")
    from datasets import Dataset
    
    # Tiny dataset
    data = [{
        "query": "Write a function to add two numbers",
        "test_input": "add(2, 3)",
        "expected_output": "5"
    }]
    dataset = Dataset.from_list(data)
    
    # Initialize trainer
    trainer = RLAIFTrainer("microsoft/DialoGPT-small")
    
    # Train for 1 episode
    history = trainer.train(dataset, num_episodes=1)
    
    print("\n✨ Test completed!")