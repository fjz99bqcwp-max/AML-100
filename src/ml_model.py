"""
ML Model Module
Hybrid LSTM + DQN Model for HFT Signal Generation
Optimized for Apple M4 with MPS (Metal Performance Shaders) acceleration
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.profiler import profile, record_function, ProfilerActivity

# Step 5: Import reward visualizer for monitoring
try:
    from monitoring.reward_visualizer import get_visualizer, RewardVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

logger = logging.getLogger(__name__)


# ==================== Model Architecture ====================

class LSTMFeatureExtractor(nn.Module):
    """
    LSTM network for temporal feature extraction from market data
    Captures sequential patterns in price/volume data
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = h_n[-1]  # (batch, hidden_size)
        out = self.layer_norm(out)
        
        return out


class DQNHead(nn.Module):
    """
    Deep Q-Network head for action value estimation
    Outputs Q-values for: Hold, Buy, Sell
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 3
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ActorCriticHead(nn.Module):
    """
    Step 4: Actor-Critic head for A2C/PPO hybrid.
    
    Actor: Outputs action probabilities (policy)
    Critic: Outputs state value estimate (value function)
    
    Combined with LSTM feature extractor for temporal patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 3
    ):
        super().__init__()
        
        # Shared layers
        self.shared_fc = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy network)
        self.actor_fc = nn.Linear(hidden_size, num_actions)
        
        # Critic head (value network)
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_probs: Softmax probabilities for each action (batch, num_actions)
            state_value: Estimated value of the state (batch, 1)
        """
        shared = F.relu(self.shared_fc(x))
        shared = self.dropout(shared)
        shared = F.relu(self.shared_fc2(shared))
        shared = self.dropout(shared)
        
        # Actor output: action probabilities
        action_logits = self.actor_fc(shared)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output: state value
        state_value = self.critic_fc(shared)
        
        return action_probs, state_value
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and return value estimate.
        
        Returns:
            action: Sampled or argmax action
            log_prob: Log probability of the action
            entropy: Policy entropy (for exploration bonus)
            value: State value estimate
        """
        action_probs, value = self.forward(x)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class HybridLSTMActorCritic(nn.Module):
    """
    Step 4: Hybrid LSTM + Actor-Critic model for A2C/PPO training.
    Uses LSTM for temporal feature extraction with PPO policy gradient.
    """
    
    def __init__(
        self,
        input_size: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        ac_hidden: int = 256,
        num_actions: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        self.ac_head = ActorCriticHead(
            input_size=lstm_hidden,
            hidden_size=ac_hidden,
            num_actions=num_actions
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action probabilities and state value"""
        features = self.lstm_extractor(x)
        return self.ac_head(features)
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value from state"""
        features = self.lstm_extractor(x)
        return self.ac_head.get_action_and_value(features, deterministic)
    
    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions for PPO loss computation.
        
        Returns:
            log_probs: Log probabilities of the actions
            values: State value estimates
            entropy: Policy entropy
        """
        features = self.lstm_extractor(x)
        action_probs, values = self.ac_head(features)
        
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class HybridLSTMDQN(nn.Module):
    """
    Hybrid LSTM + DQN model for HFT signal generation
    Combines temporal feature extraction with reinforcement learning
    """
    
    def __init__(
        self,
        input_size: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dqn_hidden: int = 256,
        num_actions: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm_extractor = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        self.dqn_head = DQNHead(
            input_size=lstm_hidden,
            hidden_size=dqn_hidden,
            num_actions=num_actions
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.lstm_extractor(x)
        q_values = self.dqn_head(features)
        return q_values
    
    def get_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[int, float]:
        """Get action with epsilon-greedy exploration"""
        if random.random() < epsilon:
            action = random.randint(0, 2)
            return action, 0.0
        
        with torch.no_grad():
            q_values = self.forward(state)
            action = int(q_values.argmax(dim=1).item())
            confidence = float(F.softmax(q_values, dim=1)[0, action].item())
            
        return action, confidence


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class RewardConfig:
    """Configuration for reward calculation - Step 2 Enhanced"""
    alpha_sharpe: float = 0.7  # Weight for Sharpe-like component
    beta_trade_penalty: float = 0.01  # Penalty for flip-flopping trades
    early_bonus_epochs: int = 8  # Epochs to apply early bonus
    early_bonus_amount: float = 0.35  # Bonus for trading in early epochs
    hold_penalty: float = 0.008  # Penalty for holding (anti-stagnation)
    commission_rate: float = 0.0005  # Transaction cost
    min_trade_reward: float = -0.3  # Floor for trade rewards (tighter)
    max_trade_reward: float = 2.0  # Ceiling for trade rewards (higher)
    diversity_penalty: float = 0.05  # Penalty when no trades in batch
    correct_direction_bonus: float = 0.45  # Bonus for correct direction
    log_scale_threshold: float = 0.3  # Apply log scaling beyond this
    trade_bonus: float = 0.03  # Bonus for any trade (encourages action)
    positive_bias: float = 0.15  # Positive offset to shift rewards upward
    vol_window: int = 20  # Window for volatility calculation
    kelly_weight: float = 0.25  # Kelly criterion weight for reward scaling


@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    epoch: int
    loss: float
    q_value_mean: float
    epsilon: float
    reward_mean: float
    validation_accuracy: float


class MLModel:
    """
    High-level ML model manager for HFT
    Handles training, inference, and model management
    """
    
    # Action mapping
    HOLD = 0
    BUY = 1
    SELL = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    def __init__(
        self,
        config_path: str = "config/params.json",
        model_dir: str = "models"
    ):
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.ml_config = self.config.get("ml_model", {})
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device (prefer MPS on M4 Mac)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for computation")
        
        # Model components (initialized later)
        self.model: Optional[HybridLSTMDQN] = None
        self.target_model: Optional[HybridLSTMDQN] = None
        self.optimizer: Optional[optim.Adam] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        
        # Mixed precision (AMP) for MPS/CUDA acceleration
        # Note: MPS has limited AMP support, use float32 fallback
        self.use_amp = self.device.type in ('cuda', 'mps')
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        
        # Early stopping and checkpointing
        self.early_stop_patience = self.ml_config.get("early_stop_patience", 15)
        self.min_reward_delta = self.ml_config.get("min_reward_delta", 0.05)
        self.best_reward_ever = float("-inf")
        self.patience_counter = 0
        
        # Reward configuration with enhanced trade bonus and Kelly weight
        reward_config = self.ml_config.get("reward", {})
        self.reward_config = RewardConfig(
            alpha_sharpe=reward_config.get("alpha_sharpe", 0.7),
            beta_trade_penalty=reward_config.get("beta_trade_penalty", 0.01),
            early_bonus_epochs=reward_config.get("early_bonus_epochs", 8),
            early_bonus_amount=reward_config.get("early_bonus_amount", 0.35),
            hold_penalty=reward_config.get("hold_penalty", 0.008),
            commission_rate=reward_config.get("commission_rate", 0.0005),
            min_trade_reward=reward_config.get("min_trade_reward", -0.3),
            max_trade_reward=reward_config.get("max_trade_reward", 2.0),
            diversity_penalty=reward_config.get("diversity_penalty", 0.05),
            correct_direction_bonus=reward_config.get("correct_direction_bonus", 0.45),
            log_scale_threshold=reward_config.get("log_scale_threshold", 0.3),
            trade_bonus=reward_config.get("trade_bonus", 0.03),
            positive_bias=reward_config.get("positive_bias", 0.15),
            vol_window=reward_config.get("vol_window", 20),
            kelly_weight=reward_config.get("kelly_weight", 0.25)
        )
        
        # Reward tracking for Sharpe-like calculation
        self.recent_rewards: deque = deque(maxlen=100)
        self.current_epoch = 0
        self.cumulative_position = 0.0  # Track net position for consecutive trade penalty
        
        # Profiling support
        self.profiling_enabled = False
        self.epoch_times: List[float] = []

        
        # Training state
        self.input_size: int = 0
        self.sequence_length = self.ml_config.get("sequence_length", 60)
        self.epsilon = self.ml_config.get("epsilon_start", 1.0)
        self.epsilon_end = self.ml_config.get("epsilon_end", 0.01)
        self.epsilon_decay = self.ml_config.get("epsilon_decay", 0.995)
        self.gamma = self.ml_config.get("gamma", 0.99)
        
        # Supervised guidance (Step 2)
        self.supervised_gamma = self.ml_config.get("supervised_gamma", 0.1)
        self.use_supervised_guidance = self.ml_config.get("use_supervised_guidance", True)
        
        # FGSM adversarial training (Step 5)
        self.use_fgsm = self.ml_config.get("use_fgsm", True)
        self.fgsm_weight = self.ml_config.get("fgsm_weight", 0.1)
        self.fgsm_epsilon = self.ml_config.get("fgsm_epsilon", 0.01)
        
        # Step 4: A2C/PPO configuration
        self.rl_algo = self.ml_config.get("rl_algo", "DQN")  # "DQN" or "A2C"
        self.ac_model: Optional[HybridLSTMActorCritic] = None
        self.ppo_clip_epsilon = self.ml_config.get("ppo_clip_epsilon", 0.2)
        self.entropy_coef = self.ml_config.get("entropy_coef", 0.01)
        self.value_loss_coef = self.ml_config.get("value_loss_coef", 0.5)
        self.gae_lambda = self.ml_config.get("gae_lambda", 0.95)
        self.ppo_epochs = self.ml_config.get("ppo_epochs", 4)  # Mini-epochs per update
        
        # Feature columns for training
        self.feature_columns: List[str] = []
        
        # Metrics
        self.training_history: List[TrainingMetrics] = []
        
    def initialize_model(self, input_size: int) -> None:
        """Initialize model architecture"""
        self.input_size = input_size
        
        self.model = HybridLSTMDQN(
            input_size=input_size,
            lstm_hidden=self.ml_config.get("lstm_hidden_size", 128),
            lstm_layers=self.ml_config.get("lstm_num_layers", 2),
            dqn_hidden=self.ml_config.get("dqn_hidden_size", 256),
            num_actions=3,
            dropout=self.ml_config.get("dropout", 0.2)
        ).to(self.device)
        
        # Target network for stable Q-learning
        self.target_model = HybridLSTMDQN(
            input_size=input_size,
            lstm_hidden=self.ml_config.get("lstm_hidden_size", 128),
            lstm_layers=self.ml_config.get("lstm_num_layers", 2),
            dqn_hidden=self.ml_config.get("dqn_hidden_size", 256),
            num_actions=3,
            dropout=self.ml_config.get("dropout", 0.2)
        ).to(self.device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.ml_config.get("learning_rate", 0.001)
        )
        
        # Learning rate scheduler - less aggressive to allow continued learning
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.ml_config.get("lr_scheduler_factor", 0.5),  # Gentler drop
            patience=self.ml_config.get("lr_scheduler_patience", 5),  # More patience
            min_lr=1e-4  # Higher min_lr to keep learning
        )
        
        self.replay_buffer = ReplayBuffer(
            capacity=self.ml_config.get("memory_size", 10000)
        )
        
        # Reset early stopping state
        self.best_reward_ever = float("-inf")
        self.patience_counter = 0
        self.epoch_times = []
        
        # Step 2: Reward threshold stopping
        self.reward_threshold_stop = self.ml_config.get("reward_threshold_stop", -40)
        self.reward_threshold_epochs = self.ml_config.get("reward_threshold_epochs", 10)
        
        logger.info(f"Model initialized with input_size={input_size}")
    
    def initialize_a2c_model(self, input_size: int) -> None:
        """
        Step 4: Initialize A2C/PPO Actor-Critic model.
        
        This is an alternative to DQN that uses policy gradients with
        an Actor (policy) and Critic (value function) architecture.
        """
        self.input_size = input_size
        
        self.ac_model = HybridLSTMActorCritic(
            input_size=input_size,
            lstm_hidden=self.ml_config.get("lstm_hidden_size", 128),
            lstm_layers=self.ml_config.get("lstm_num_layers", 2),
            ac_hidden=self.ml_config.get("dqn_hidden_size", 256),
            num_actions=3,
            dropout=self.ml_config.get("dropout", 0.2)
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.ac_model.parameters(),
            lr=self.ml_config.get("learning_rate", 0.001)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # A2C doesn't use replay buffer, but keep for compatibility
        self.replay_buffer = ReplayBuffer(
            capacity=self.ml_config.get("memory_size", 10000)
        )
        
        # Reset early stopping state
        self.best_reward_ever = float("-inf")
        self.patience_counter = 0
        self.epoch_times = []
        
        logger.info(f"A2C/PPO model initialized with input_size={input_size}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare feature matrix from DataFrame - Step 3 v2 with expanded features"""
        # Select numeric columns for features (expanded from 23 to 35+)
        feature_cols = [
            # Price action
            "returns", "log_returns",
            # Moving averages
            "sma_7", "sma_14", "sma_21", "sma_50",
            "ema_7", "ema_14", "ema_21", "ema_50",
            # Core indicators
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_width", "bb_position",
            "atr_pct", "volume_ratio",
            # Momentum
            "momentum_5", "momentum_10", "momentum_20",
            "volatility", "adx",
            # Step 3 Enhanced: Price patterns
            "higher_high", "lower_low", "higher_close",
            "pv_divergence", "body_size", "upper_shadow", "lower_shadow",
            "price_position_5", "price_position_20", "roc_5", "roc_10",
            # Step 3 v2: Orderbook-like features
            "bid_ask_spread_pct", "bid_side_pressure", "ask_side_pressure",
            "imbalance_ratio", "depth_ratio",
            # Step 3 v2: Funding and regime
            "funding_proxy", "zscore_20", "zscore_50", "vol_regime"
        ]
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = available_cols
        
        # Normalize features
        features = df[available_cols].values
        
        # Handle any remaining NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std
        
        return features, available_cols
    
    def create_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        atr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM training"""
        seq_len = self.sequence_length
        n_samples = len(features) - seq_len - 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data for sequence length {seq_len}")
        
        X = np.zeros((n_samples, seq_len, features.shape[1]))
        y_prices = np.zeros(n_samples)
        current_prices = np.zeros(n_samples)
        atr_values = np.zeros(n_samples) if atr is not None else None
        
        for i in range(n_samples):
            X[i] = features[i:i + seq_len]
            current_prices[i] = prices[i + seq_len - 1]
            y_prices[i] = prices[i + seq_len]
            if atr is not None and atr_values is not None:
                atr_values[i] = atr[i + seq_len - 1]
        
        return X, current_prices, y_prices, atr_values
    
    def calculate_reward(
        self,
        action: int,
        price_change_pct: float,
        atr_normalized: float = 1.0,
        prev_action: Optional[int] = None,
        status: str = "normal"
    ) -> float:
        """
        Refined reward function for positive rewards and trade encouragement (Step 2).
        
        Key improvements:
        - Trade bonus for any non-HOLD action (encourage activity)
        - Positive bias to shift overall rewards toward positive
        - Kelly-normalized returns using log(1 + return)
        - Stronger correct direction bonus
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price_change_pct: (future_price - current_price) / current_price
            atr_normalized: ATR / price for volatility normalization (default 1.0)
            prev_action: Previous action to detect consecutive trades
            status: Training status (Critical/Poor/Normal) for scaling
        
        Returns:
            Calculated reward (biased positive for good trades)
        """
        cfg = self.reward_config
        
        # Status-based scaling (Critical retrains boost exploration)
        is_critical = status.lower() == "critical"
        status_scale = 1.5 if is_critical else 1.0
        
        # TRADE BONUS: Encourage any trading action (Step 2 enhancement)
        trade_bonus = 0.0
        if action != self.HOLD:
            trade_bonus = cfg.trade_bonus
            # Extra bonus in critical mode to break stagnation
            if is_critical and self.current_epoch < cfg.early_bonus_epochs:
                trade_bonus += cfg.trade_bonus * 1.0  # Step 3: Full bonus in critical
        
        # Base reward from price movement using log(1 + return) for stability
        if action == self.HOLD:
            # HOLD dominance penalty: punish excessive inactivity
            # Track consecutive holds to escalate penalty
            if not hasattr(self, '_consecutive_holds'):
                self._consecutive_holds = 0
            
            self._consecutive_holds += 1
            
            # Escalating penalty for repeated holds
            hold_multiplier = 1.0 + (self._consecutive_holds * 0.1)  # 10% increase per consecutive hold
            missed_opportunity = abs(price_change_pct) * 0.8
            base_reward = -(cfg.hold_penalty + missed_opportunity) * status_scale * 1.5 * hold_multiplier
        elif action == self.BUY:
            # Reset consecutive holds on trade
            self._consecutive_holds = 0
            
            # Long position: profit from price increase
            # Use log scaling for more stable gradients
            if price_change_pct > 0:
                base_reward = np.log1p(price_change_pct * 100) / 100 - cfg.commission_rate
            else:
                base_reward = price_change_pct - cfg.commission_rate
        else:  # SELL
            # Reset consecutive holds on trade
            self._consecutive_holds = 0
            
            # Short position: profit from price decrease  
            if price_change_pct < 0:
                base_reward = np.log1p(-price_change_pct * 100) / 100 - cfg.commission_rate
            else:
                base_reward = -price_change_pct - cfg.commission_rate
        
        # Kelly-weighted returns (Step 2: incorporate Kelly criterion)
        kelly_multiplier = 1.0 + cfg.kelly_weight * (1.0 if base_reward > 0 else 0.5)
        base_reward = base_reward * kelly_multiplier
        
        # Normalize by ATR (volatility-adjusted returns) with floor
        atr_factor = max(atr_normalized, 0.001)
        base_reward = base_reward / atr_factor
        
        # Step 3: VOLATILITY REGIME BONUS - reward trades in high volatility
        vol_regime_bonus = 0.0
        if atr_normalized > 0.01 and action != self.HOLD:  # High volatility
            vol_regime_bonus = 0.05  # Bonus for trading in volatile markets
        
        # CORRECT DIRECTION BONUS: Stronger reward for right direction
        direction_bonus = 0.0
        if action == self.BUY and price_change_pct > 0.0001:
            direction_bonus = cfg.correct_direction_bonus * 1.2  # Step 3: 20% boost
        elif action == self.SELL and price_change_pct < -0.0001:
            direction_bonus = cfg.correct_direction_bonus * 1.2  # Step 3: 20% boost
        elif action != self.HOLD:
            # Wrong direction: smaller penalty
            direction_bonus = -cfg.correct_direction_bonus * 0.2  # Step 3: Reduced penalty
        
        # Sharpe-like component: reward / recent_volatility
        sharpe_bonus = 0.0
        if len(self.recent_rewards) >= 5:
            recent_std = np.std(list(self.recent_rewards)) + 1e-8
            if action != self.HOLD and base_reward > 0:
                sharpe_bonus = cfg.alpha_sharpe * (base_reward / recent_std)
        
        # Early epoch trade bonus (encourage exploration)
        early_bonus = 0.0
        if self.current_epoch < cfg.early_bonus_epochs and action != self.HOLD:
            early_bonus = cfg.early_bonus_amount * 1.2  # Step 3: 20% boost
            # BUY BIAS: In early epochs, add small bonus for BUY to balance SELL dominance
            if action == self.BUY:
                early_bonus += 0.03  # Step 3: Increased from 0.02
        
        # BUY BALANCE BONUS: During critical retrains, prefer under-represented action
        buy_balance_bonus = 0.0
        if is_critical and action == self.BUY:
            # Critical mode often has SELL bias - small BUY bonus to rebalance
            buy_balance_bonus = 0.02  # Step 3: Increased from 0.015
        
        # Consecutive trade penalty (reduced to not discourage trading)
        trade_penalty = 0.0
        if prev_action is not None and prev_action != self.HOLD and action != self.HOLD:
            if prev_action != action:  # Flip-flopping only
                trade_penalty = -cfg.beta_trade_penalty * 0.5  # Step 3: Halved penalty
        
        # Step 3: Return alpha bonus - prioritize raw returns for monthly target
        # Uses log(1 + |return|) for stability, scaled by return_alpha
        return_alpha = getattr(cfg, 'return_alpha', 0.8)  # Step 3: Increased from 0.6
        return_bonus = 0.0
        if action != self.HOLD and price_change_pct != 0:
            is_profitable = (action == self.BUY and price_change_pct > 0) or \
                           (action == self.SELL and price_change_pct < 0)
            if is_profitable:
                return_bonus = return_alpha * np.log1p(abs(price_change_pct) * 100)
            else:
                return_bonus = -return_alpha * 0.2 * np.log1p(abs(price_change_pct) * 100)  # Step 3: Reduced loss penalty
        
        # Combine all components
        total_reward = (
            base_reward 
            + direction_bonus 
            + sharpe_bonus 
            + early_bonus 
            + trade_bonus 
            + trade_penalty
            + return_bonus  # Step 1: Return-weighted bonus
            + buy_balance_bonus  # Step 2: BUY bias for balance
            + vol_regime_bonus  # Step 3: Volatility trading bonus
            + cfg.positive_bias  # Positive shift to encourage positive rewards
        )
        
        # LOG-SCALING for extreme rewards
        if total_reward < -cfg.log_scale_threshold:
            total_reward = -cfg.log_scale_threshold - np.log1p(-total_reward - cfg.log_scale_threshold)
        elif total_reward > cfg.log_scale_threshold:
            total_reward = cfg.log_scale_threshold + np.log1p(total_reward - cfg.log_scale_threshold)
        
        # Clip to range (asymmetric: less negative floor, higher ceiling)
        total_reward = np.clip(total_reward, cfg.min_trade_reward, cfg.max_trade_reward)
        
        # Track for Sharpe calculation
        self.recent_rewards.append(total_reward)
        
        return float(total_reward)
    
    def calculate_optimal_q_values(
        self,
        price_changes: np.ndarray,
        macd_values: Optional[np.ndarray] = None,
        rsi_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Step 2: Compute optimal Q-values based on technical indicators.
        Used for supervised guidance in loss function.
        
        Args:
            price_changes: Array of price changes for each sample
            macd_values: MACD histogram values (optional)
            rsi_values: RSI values (optional)
            
        Returns:
            optimal_q: (n_samples, 3) tensor of optimal Q-values
        """
        n_samples = len(price_changes)
        optimal_q = np.zeros((n_samples, 3))  # [HOLD, BUY, SELL]
        
        for i in range(n_samples):
            pc = price_changes[i]
            
            # Base signal from price direction
            if pc > 0.001:  # Price going up -> BUY is optimal
                optimal_q[i] = [0.0, 1.0, -1.0]
            elif pc < -0.001:  # Price going down -> SELL is optimal
                optimal_q[i] = [-0.0, -1.0, 1.0]
            else:  # Flat -> HOLD is optimal
                optimal_q[i] = [0.5, 0.0, 0.0]
            
            # Enhance with MACD if available
            if macd_values is not None and not np.isnan(macd_values[i]):
                macd = macd_values[i]
                if macd > 0:
                    optimal_q[i, 1] += 0.3  # Boost BUY
                    optimal_q[i, 2] -= 0.3
                else:
                    optimal_q[i, 2] += 0.3  # Boost SELL
                    optimal_q[i, 1] -= 0.3
            
            # Enhance with RSI if available
            if rsi_values is not None and not np.isnan(rsi_values[i]):
                rsi = rsi_values[i]
                if rsi < 30:  # Oversold -> BUY signal
                    optimal_q[i, 1] += 0.5
                    optimal_q[i, 0] -= 0.2
                elif rsi > 70:  # Overbought -> SELL signal
                    optimal_q[i, 2] += 0.5
                    optimal_q[i, 0] -= 0.2
        
        return optimal_q
    
    def train_episode(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        status: str = "normal"
    ) -> float:
        """
        Train on one episode of data.
        Optimized with vectorized experience collection and batched training.
        
        Enhanced with:
        - ATR-normalized rewards (Step 1)
        - Supervised guidance from technical indicators (Step 2)
        - Previous action tracking for trade penalty
        - Status-based scaling for Critical retrains (Step 4)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Extract ATR for reward normalization if available
        atr_values = None
        macd_values = None
        rsi_values = None
        if df is not None:
            if "atr_pct" in df.columns:
                atr_values = df["atr_pct"].values
            if "macd_hist" in df.columns:
                macd_values = df["macd_hist"].values
            if "rsi" in df.columns:
                rsi_values = df["rsi"].values
        
        X, current_prices, future_prices, atr_seq = self.create_sequences(
            features, prices, atr_values
        )
        
        batch_size = self.ml_config.get("batch_size", 256)
        num_samples = len(X) - 1
        
        # Vectorized: Calculate all price changes at once
        price_changes = (future_prices[:-1] - current_prices[:-1]) / (current_prices[:-1] + 1e-8)
        
        # ATR for normalization
        atr_normalized = atr_seq[:-1] if atr_seq is not None else np.ones(num_samples) * 0.01
        
        # Convert all data to tensors at once (much faster than per-sample)
        all_states = torch.FloatTensor(X[:-1]).to(self.device)
        all_next_states = torch.FloatTensor(X[1:]).to(self.device)
        
        # Sample a subset for experience collection (e.g., 20% of data per epoch)
        # This dramatically reduces per-epoch time while maintaining learning
        sample_ratio = self.ml_config.get("sample_ratio", 0.2)
        num_to_sample = max(batch_size * 4, int(num_samples * sample_ratio))
        sample_indices = np.random.choice(num_samples, size=min(num_to_sample, num_samples), replace=False)
        
        total_reward = 0.0
        
        # Batch action selection (vectorized)
        self.model.eval()
        with torch.no_grad():
            sampled_states = all_states[sample_indices]
            q_values = self.model(sampled_states)
            
            if self.epsilon > np.random.random():
                # Random actions for exploration
                actions = np.random.randint(0, 3, size=len(sample_indices))
            else:
                # Greedy actions
                actions = q_values.argmax(dim=1).cpu().numpy()
        
        self.model.train()
        
        # Enhanced reward calculation with ATR normalization and status
        sampled_price_changes = price_changes[sample_indices]
        sampled_atr = atr_normalized[sample_indices]
        rewards = np.zeros(len(sample_indices))
        prev_action = None
        
        # Track action distribution for diversity enforcement
        action_counts = np.zeros(3)  # HOLD, BUY, SELL
        
        for i, (action, pc, atr) in enumerate(zip(actions, sampled_price_changes, sampled_atr)):
            rewards[i] = self.calculate_reward(
                action=action,
                price_change_pct=pc,
                atr_normalized=atr,
                prev_action=prev_action,
                status=status  # Step 4: Pass status for scaling
            )
            prev_action = action
            action_counts[action] += 1
        
        trade_count = int(action_counts[1] + action_counts[2])  # BUY + SELL
        
        # Step 3: Enhanced diversity threshold from config (lowered to 40%)
        diversity_thresh = self.ml_config.get("diversity_thresh", 0.4)
        min_action_pct = self.ml_config.get("min_action_pct", 0.30)  # Step 3: Min 30% trades
        
        # Strong diversity penalty if model is stuck on one action
        total_actions = len(actions)
        if total_actions > 10:
            max_action_pct = action_counts.max() / total_actions
            
            # Step 3: HOLD DOMINANCE - severe penalty for >60% HOLD
            hold_pct = action_counts[self.HOLD] / total_actions
            if hold_pct > 0.60:  # Step 3: HOLD > 60% threshold
                hold_penalty = -self.reward_config.diversity_penalty * 12 * total_actions  # Step 3: 12x penalty
                rewards = rewards + (hold_penalty / len(rewards))
                logger.warning(f"⚠️ HOLD dominance penalty: {hold_pct:.1%} HOLD (max 60%)")
            
            if max_action_pct > diversity_thresh:  # Configurable threshold
                diversity_penalty = -self.reward_config.diversity_penalty * 10 * total_actions  # Step 3: Increased from 8
                rewards = rewards + (diversity_penalty / len(rewards))
                logger.debug(f"Diversity penalty applied: {max_action_pct:.1%} same action (thresh={diversity_thresh:.0%})")
            
            # Step 3: MIN ACTION PCT - enforce minimum trades in critical mode
            trade_pct = trade_count / total_actions
            is_critical = status.lower() == "critical" if status else False
            if is_critical and trade_pct < min_action_pct:
                # Critical mode requires minimum trading activity
                deficit = min_action_pct - trade_pct
                action_deficit_penalty = -self.reward_config.diversity_penalty * 15 * deficit * total_actions
                rewards = rewards + (action_deficit_penalty / len(rewards))
                logger.warning(f"⚠️ Critical trade deficit: {trade_pct:.1%} trades (min {min_action_pct:.0%})")
            
            # BUY/SELL BALANCE: Force balance between BUY and SELL actions
            buy_count = action_counts[self.BUY]
            sell_count = action_counts[self.SELL]
            trade_total = buy_count + sell_count
            if trade_total > 5:
                # Penalty if BUY or SELL dominates (>65% of trades) - Step 3: Tighter balance
                buy_ratio = buy_count / trade_total
                sell_ratio = sell_count / trade_total
                if sell_ratio > 0.65:  # Step 3: Reduced from 0.7
                    # Heavy SELL bias detected - penalize to encourage BUY
                    imbalance_penalty = -self.reward_config.diversity_penalty * 8 * total_actions  # Step 3: Increased
                    rewards = rewards + (imbalance_penalty / len(rewards))
                    logger.debug(f"SELL bias penalty: {sell_ratio:.1%} SELL vs {buy_ratio:.1%} BUY")
                elif buy_ratio > 0.65:  # Step 3: Reduced from 0.7
                    # Heavy BUY bias detected - penalize to encourage SELL
                    imbalance_penalty = -self.reward_config.diversity_penalty * 8 * total_actions  # Step 3: Increased
                    rewards = rewards + (imbalance_penalty / len(rewards))
                    logger.debug(f"BUY bias penalty: {buy_ratio:.1%} BUY vs {sell_ratio:.1%} SELL")
        
        # Additional penalty if no trades at all - Step 3: Much stronger now
        if trade_count == 0 and total_actions > 10:
            diversity_penalty = -self.reward_config.diversity_penalty * total_actions * 4  # Step 3: 4x stronger
            rewards = rewards + (diversity_penalty / len(rewards))
            logger.warning(f"⚠️ No trades in episode - severe penalty applied")
        
        total_reward = rewards.sum()
        
        # Type guard for replay buffer
        if self.replay_buffer is None:
            raise RuntimeError("replay_buffer is not initialized")
        
        # Bulk push to replay buffer
        sampled_next_indices = np.minimum(sample_indices + 1, num_samples - 1)
        for i, idx in enumerate(sample_indices):
            next_idx = sampled_next_indices[i]
            done = (idx == num_samples - 1)
            self.replay_buffer.push(
                X[idx], actions[i], rewards[i], X[next_idx], done
            )
        
        # Train multiple batches per episode for better sample efficiency
        # Include supervised guidance (Step 2 + Step 4)
        num_train_batches = self.ml_config.get("train_batches_per_epoch", 10)
        
        # Prepare optimal Q-values for supervised guidance
        optimal_q = None
        if self.use_supervised_guidance:
            sampled_macd = macd_values[self.sequence_length:-1][sample_indices] if macd_values is not None else None
            sampled_rsi = rsi_values[self.sequence_length:-1][sample_indices] if rsi_values is not None else None
            optimal_q = self.calculate_optimal_q_values(
                sampled_price_changes, sampled_macd, sampled_rsi
            )
        
        # Step 4: Stronger supervised guidance for Critical status
        effective_gamma = self.supervised_gamma
        if status.lower() == "critical":
            effective_gamma = self.ml_config.get("supervised_gamma_critical", 0.5)
        
        if len(self.replay_buffer) >= batch_size:
            for _ in range(num_train_batches):
                self._train_batch(batch_size, optimal_q, effective_gamma)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Return mean reward per sample for consistent comparison
        mean_reward = total_reward / len(sample_indices) if len(sample_indices) > 0 else 0.0
        return mean_reward
    
    def _train_batch(
        self,
        batch_size: int,
        optimal_q: Optional[np.ndarray] = None,
        supervised_gamma_override: Optional[float] = None
    ) -> float:
        """
        Train on a batch from replay buffer with AMP support.
        
        Step 2 + Step 4: Includes supervised guidance loss from optimal Q-values
        based on technical indicators (MACD, RSI). Uses stronger guidance
        for Critical retrains via supervised_gamma_override.
        """
        # Type guards
        if self.replay_buffer is None:
            raise RuntimeError("replay_buffer is not initialized")
        if self.model is None or self.target_model is None:
            raise RuntimeError("models are not initialized")
        if self.optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        
        # Use override gamma if provided (Step 4: Critical status uses higher gamma)
        effective_gamma = supervised_gamma_override if supervised_gamma_override is not None else self.supervised_gamma
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Determine autocast device type
        autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        # Forward pass with optional AMP
        with autocast(device_type=autocast_device, enabled=self.use_amp and self.device.type == 'cuda'):
            # Current Q values (all actions)
            all_q = self.model(states)
            current_q = all_q.gather(1, actions.unsqueeze(1))
            
            # Target Q values (Double DQN)
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1, keepdim=True)
                next_q = self.target_model(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (~dones).unsqueeze(1)
            
            # Primary loss: TD error (Huber loss)
            td_loss = F.smooth_l1_loss(current_q, target_q)
            
            # Step 2 + Step 4: Supervised guidance loss with effective gamma
            supervised_loss = torch.tensor(0.0, device=self.device)
            if self.use_supervised_guidance and optimal_q is not None:
                # Sample random indices from optimal_q for this batch
                opt_indices = np.random.choice(
                    len(optimal_q), 
                    size=min(batch_size, len(optimal_q)), 
                    replace=False
                )
                opt_q_batch = torch.FloatTensor(optimal_q[opt_indices]).to(self.device)
                
                # Use Q-values from random states in batch to match
                model_q_sample = all_q[:len(opt_indices)]
                
                # MSE loss between model Q and optimal Q (teacher signal)
                supervised_loss = F.mse_loss(model_q_sample, opt_q_batch)
            
            # Combined loss with effective gamma (Step 4: higher for Critical)
            loss = td_loss + effective_gamma * supervised_loss
            
            # Step 2: Add entropy bonus to encourage exploration and diverse actions
            entropy_bonus_coef = self.ml_config.get("entropy_bonus", 0.01)
            if entropy_bonus_coef > 0:
                # Compute policy entropy from Q-values (softmax distribution)
                q_probs = F.softmax(all_q, dim=1)
                q_log_probs = F.log_softmax(all_q, dim=1)
                entropy = -(q_probs * q_log_probs).sum(dim=1).mean()
                # Subtract entropy (maximize entropy = minimize negative entropy)
                loss = loss - entropy_bonus_coef * entropy
            
            # Step 5: FGSM adversarial training for robustness
            if self.use_fgsm and random.random() < 0.3:  # Apply 30% of the time
                adv_loss = self._fgsm_adversarial_loss(states, actions, target_q)
                loss = loss + self.fgsm_weight * adv_loss
        
        # Backward pass with gradient scaling (CUDA only)
        self.optimizer.zero_grad()
        
        if self.device.type == 'cuda' and self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
    
    def _fgsm_adversarial_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_q: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """
        Step 5: Fast Gradient Sign Method (FGSM) adversarial training.
        
        Perturbs inputs to create adversarial examples, then trains
        the model to be robust against these perturbations.
        Simulates BTC price spikes/volatility.
        
        Args:
            states: Input state tensors (batch, seq, features)
            actions: Action indices
            target_q: Target Q-values
            epsilon: Perturbation magnitude
            
        Returns:
            Adversarial loss component
        """
        if self.model is None:
            return torch.tensor(0.0, device=self.device)
        
        # Enable gradients for input
        states_adv = states.clone().detach().requires_grad_(True)
        
        # Forward pass
        q_values = self.model(states_adv)
        current_q = q_values.gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Get gradient of loss w.r.t. input
        loss.backward(retain_graph=True)
        
        # Create adversarial perturbation (FGSM)
        if states_adv.grad is not None:
            perturbation = epsilon * states_adv.grad.sign()
            states_perturbed = states_adv + perturbation
            
            # Clamp to valid range (normalized features typically in [-3, 3])
            states_perturbed = torch.clamp(states_perturbed, -3.0, 3.0)
            
            # Forward pass on perturbed input
            q_values_adv = self.model(states_perturbed.detach())
            current_q_adv = q_values_adv.gather(1, actions.unsqueeze(1))
            
            # Adversarial loss: model should still predict correctly
            adv_loss = F.smooth_l1_loss(current_q_adv, target_q)
            
            return adv_loss
        
        return torch.tensor(0.0, device=self.device)
    
    def train_a2c_episode(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Step 4: A2C/PPO training episode.
        
        Uses policy gradients with advantage estimation for more stable
        training compared to DQN. Better suited for continuous action
        spaces and fine-grained trading decisions.
        """
        if self.ac_model is None:
            raise ValueError("A2C model not initialized. Call initialize_a2c_model first.")
        
        # Extract ATR for reward normalization if available
        atr_values = None
        if df is not None and "atr_pct" in df.columns:
            atr_values = df["atr_pct"].values
        
        X, current_prices, future_prices, atr_seq = self.create_sequences(
            features, prices, atr_values
        )
        
        num_samples = len(X) - 1
        
        # Vectorized price changes
        price_changes = (future_prices[:-1] - current_prices[:-1]) / (current_prices[:-1] + 1e-8)
        atr_normalized = atr_seq[:-1] if atr_seq is not None else np.ones(num_samples) * 0.01
        
        # Convert all data to tensors
        all_states = torch.FloatTensor(X[:-1]).to(self.device)
        
        # Sample subset for training
        sample_ratio = self.ml_config.get("sample_ratio", 0.2)
        batch_size = self.ml_config.get("batch_size", 256)
        num_to_sample = max(batch_size, int(num_samples * sample_ratio))
        sample_indices = np.random.choice(num_samples, size=min(num_to_sample, num_samples), replace=False)
        
        sampled_states = all_states[sample_indices]
        sampled_price_changes = price_changes[sample_indices]
        sampled_atr = atr_normalized[sample_indices]
        
        # Collect trajectory using current policy
        self.ac_model.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.ac_model.get_action_and_value(
                sampled_states, deterministic=False
            )
        
        self.ac_model.train()
        
        # Calculate rewards
        actions_np = actions.cpu().numpy()
        rewards = np.zeros(len(sample_indices))
        prev_action = None
        
        for i, (action, pc, atr) in enumerate(zip(actions_np, sampled_price_changes, sampled_atr)):
            rewards[i] = self.calculate_reward(
                action=action,
                price_change_pct=pc,
                atr_normalized=atr,
                prev_action=prev_action
            )
            prev_action = action
        
        total_reward = rewards.sum()
        
        # Convert rewards to tensor
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards_t, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        returns = advantages + values.detach()
        
        for _ in range(self.ppo_epochs):
            self._ppo_update(
                sampled_states, actions, log_probs.detach(),
                returns, advantages.detach()
            )
        
        return total_reward
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides lower variance advantage estimates compared to
        simple TD(0) or Monte Carlo returns.
        """
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        last_gae = 0.0
        
        # We don't have next values in this simple version, use 0
        next_value = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()
            
            # TD error
            delta = rewards[t] + self.gamma * next_val - values[t]
            
            # GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * last_gae
            last_gae = advantages[t].item()
        
        return advantages
    
    def _ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> float:
        """
        PPO (Proximal Policy Optimization) update step.
        
        Uses clipped objective to prevent large policy updates that
        could destabilize training.
        """
        if self.ac_model is None or self.optimizer is None:
            raise RuntimeError("A2C model or optimizer not initialized")
        
        # Get new action probabilities and values
        new_log_probs, values, entropy = self.ac_model.evaluate_actions(states, actions)
        
        # Policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network with current model weights"""
        if self.target_model is not None and self.model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
    
    async def train(
        self,
        df: pd.DataFrame,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with optimizations:
        - Early stopping with patience
        - Learning rate scheduling
        - Reward-based checkpointing
        - Epoch timing and profiling support
        
        Step 4: Routes to A2C training when rl_algo='A2C' is configured.
        """
        # Route to A2C training if configured
        if self.rl_algo.upper() == "A2C":
            return await self.train_a2c(df, epochs)
        
        # Original DQN training
        if epochs is None:
            epochs = self.ml_config.get("epochs", 100)
        
        logger.info(f"Starting DQN training for {epochs} epochs")
        logger.info(f"Early stopping: patience={self.early_stop_patience}, min_delta={self.min_reward_delta}")
        
        # Prepare features
        features, feature_cols = self.prepare_features(df)
        prices = df["close"].values
        
        # Initialize model if needed
        if self.model is None:
            self.initialize_model(len(feature_cols))
        
        # Ensure replay buffer is initialized (may be None if model was loaded)
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                capacity=self.ml_config.get("memory_size", 10000)
            )
        
        # Ensure scheduler is initialized
        if self.scheduler is None:
            if self.optimizer is None:
                raise RuntimeError("optimizer is not initialized")
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        
        target_update_freq = self.ml_config.get("target_update_freq", 10)
        
        # Type guards for required components
        if epochs is None:
            epochs = 50  # Default fallback
        if self.optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        
        # Training state
        best_reward = float("-inf")
        training_start = time.time()
        self.epoch_times = []
        self.patience_counter = 0
        epochs_completed = 0
        early_stopped = False
        
        # Get current learning rate for logging
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Initial learning rate: {current_lr:.6f}")
        
        # Reset reward tracking for new training session
        self.recent_rewards.clear()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            self.current_epoch = epoch  # Track for early bonus in reward
            
            # Train episode with DataFrame for feature extraction
            epoch_reward = self.train_episode(features, prices, df)
            
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            epochs_completed = epoch + 1
            
            # Update target network periodically
            if epochs_completed % target_update_freq == 0:
                self.update_target_network()
            
            # Update LR scheduler based on reward
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(epoch_reward)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    logger.info(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Early stopping and checkpointing
            if epoch_reward > best_reward + self.min_reward_delta:
                best_reward = epoch_reward
                self.save_model("best_model.pt")
                self.patience_counter = 0
                logger.info(f"New best reward: {best_reward:.4f} - Model saved")
            else:
                self.patience_counter += 1
            
            # Log progress every 10 epochs or on improvement
            if epochs_completed % 10 == 0 or self.patience_counter == 0:
                avg_epoch_time = np.mean(self.epoch_times[-10:]) if self.epoch_times else epoch_time
                remaining_epochs = epochs - epochs_completed
                eta_seconds = remaining_epochs * avg_epoch_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch {epochs_completed}/{epochs} - "
                    f"Reward: {epoch_reward:.4f}, "
                    f"Best: {best_reward:.4f}, "
                    f"Epsilon: {self.epsilon:.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"ETA: {eta_seconds/60:.1f}min"
                )
            
            # Check early stopping (patience-based)
            if self.patience_counter >= self.early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epochs_completed} "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                early_stopped = True
                break
            
            # Step 2: Reward threshold check - stop if still very negative after N epochs
            reward_threshold = self.ml_config.get("reward_threshold_stop", -40)
            threshold_epochs = self.ml_config.get("reward_threshold_epochs", 10)
            if epochs_completed >= threshold_epochs and best_reward < reward_threshold:
                logger.warning(
                    f"Reward threshold stop at epoch {epochs_completed} "
                    f"(best {best_reward:.4f} < threshold {reward_threshold})"
                )
                early_stopped = True
                break
            
            # Allow other async tasks to run
            await asyncio.sleep(0)
        
        training_time = time.time() - training_start
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        results = {
            "epochs": epochs_completed,
            "epochs_requested": epochs,
            "best_reward": best_reward,
            "final_epsilon": self.epsilon,
            "training_time": training_time,
            "avg_epoch_time": avg_epoch_time,
            "feature_count": len(feature_cols),
            "sample_count": len(df),
            "early_stopped": early_stopped,
            "final_lr": self.optimizer.param_groups[0]['lr']
        }
        
        # Save final model
        self.save_model("final_model.pt")
        
        logger.info(
            f"Training complete in {training_time:.2f}s "
            f"({epochs_completed} epochs, {avg_epoch_time:.2f}s/epoch avg)"
        )
        
        if early_stopped:
            logger.info(f"Saved {epochs - epochs_completed} epochs via early stopping")
        
        # Step 5: Save reward visualization for DQN training
        if HAS_VISUALIZER:
            try:
                viz = get_visualizer()
                # Add epoch rewards for visualization
                for r in list(self.recent_rewards):
                    viz.add_rewards([r], 0)
                viz.epoch_rewards = list(self.recent_rewards)[-epochs_completed:] if len(self.recent_rewards) > 0 else []
                
                hist_path = viz.save_histogram(
                    filename="reward_hist.png",
                    title=f"DQN Training - Best: {best_reward:.4f}, Epochs: {epochs_completed}"
                )
                if hist_path:
                    logger.info(f"Reward histogram saved to {hist_path}")
                    results["visualization_path"] = hist_path
                
                json_path = viz.save_json("reward_data.json")
                results["reward_data_path"] = json_path
            except Exception as e:
                logger.warning(f"Failed to save reward visualization: {e}")
        
        return results
    
    async def train_a2c(
        self,
        df: pd.DataFrame,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Step 4: Full A2C/PPO training loop.
        
        Uses Actor-Critic architecture with PPO updates for more
        stable policy optimization. Better for fine-grained trading.
        """
        if epochs is None:
            epochs = self.ml_config.get("epochs", 100)
        
        logger.info(f"Starting A2C/PPO training for {epochs} epochs")
        logger.info(f"PPO clip: {self.ppo_clip_epsilon}, GAE lambda: {self.gae_lambda}")
        logger.info(f"Early stopping: patience={self.early_stop_patience}, min_delta={self.min_reward_delta}")
        
        # Prepare features
        features, feature_cols = self.prepare_features(df)
        prices = df["close"].values
        
        # Initialize A2C model if needed
        if self.ac_model is None:
            self.initialize_a2c_model(len(feature_cols))
        
        # Ensure scheduler is initialized
        if self.scheduler is None and self.optimizer is not None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        
        # Type guards
        if epochs is None:
            epochs = 50
        if self.optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        
        # Training state
        best_reward = float("-inf")
        training_start = time.time()
        self.epoch_times = []
        self.patience_counter = 0
        epochs_completed = 0
        early_stopped = False
        
        # Get current learning rate for logging
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Initial learning rate: {current_lr:.6f}")
        
        # Reset reward tracking for new training session
        self.recent_rewards.clear()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            self.current_epoch = epoch
            
            # Train A2C episode
            epoch_reward = self.train_a2c_episode(features, prices, df)
            
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            epochs_completed = epoch + 1
            
            # Update LR scheduler based on reward
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(epoch_reward)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    logger.info(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Early stopping and checkpointing
            if epoch_reward > best_reward + self.min_reward_delta:
                best_reward = epoch_reward
                self.save_a2c_model("best_a2c_model.pt")
                self.patience_counter = 0
                logger.info(f"New best reward: {best_reward:.4f} - A2C Model saved")
            else:
                self.patience_counter += 1
            
            # Log progress every 10 epochs or on improvement
            if epochs_completed % 10 == 0 or self.patience_counter == 0:
                avg_epoch_time = np.mean(self.epoch_times[-10:]) if self.epoch_times else epoch_time
                remaining_epochs = epochs - epochs_completed
                eta_seconds = remaining_epochs * avg_epoch_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"[A2C] Epoch {epochs_completed}/{epochs} - "
                    f"Reward: {epoch_reward:.4f}, "
                    f"Best: {best_reward:.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"ETA: {eta_seconds/60:.1f}min"
                )
            
            # Check early stopping
            if self.patience_counter >= self.early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epochs_completed} "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                early_stopped = True
                break
            
            # Allow other async tasks to run
            await asyncio.sleep(0)
        
        training_time = time.time() - training_start
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        results = {
            "algorithm": "A2C/PPO",
            "epochs": epochs_completed,
            "epochs_requested": epochs,
            "best_reward": best_reward,
            "training_time": training_time,
            "avg_epoch_time": avg_epoch_time,
            "feature_count": len(feature_cols),
            "sample_count": len(df),
            "early_stopped": early_stopped,
            "final_lr": self.optimizer.param_groups[0]['lr']
        }
        
        # Save final A2C model
        self.save_a2c_model("final_a2c_model.pt")
        
        logger.info(
            f"A2C training complete in {training_time:.2f}s "
            f"({epochs_completed} epochs, {avg_epoch_time:.2f}s/epoch avg)"
        )
        
        if early_stopped:
            logger.info(f"Saved {epochs - epochs_completed} epochs via early stopping")
        
        return results
    
    def save_a2c_model(self, filename: str) -> None:
        """Save A2C model checkpoint"""
        if self.ac_model is None:
            return
        
        path = self.model_dir / filename
        torch.save({
            'model_state': self.ac_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'input_size': self.input_size,
            'feature_columns': self.feature_columns,
            'rl_algo': 'A2C'
        }, path)
        logger.debug(f"A2C model saved to {path}")
    
    def load_a2c_model(self, filename: str) -> None:
        """Load A2C model checkpoint"""
        path = self.model_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        input_size = checkpoint['input_size']
        
        self.initialize_a2c_model(input_size)
        
        if self.ac_model is not None:
            self.ac_model.load_state_dict(checkpoint['model_state'])
        
        if self.optimizer and checkpoint.get('optimizer_state'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.feature_columns = checkpoint.get('feature_columns', [])
        logger.info(f"A2C model loaded from {path}")
    
    def predict_a2c(
        self,
        features: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float, np.ndarray]:
        """
        Predict action using A2C model.
        Returns: (action, confidence, action_probs)
        """
        if self.ac_model is None:
            raise ValueError("A2C model not initialized")
        
        self.ac_model.eval()
        
        with torch.no_grad():
            if len(features.shape) == 2:
                features = features[np.newaxis, :]
            
            state = torch.FloatTensor(features).to(self.device)
            action_probs, value = self.ac_model(state)
            
            if deterministic:
                action = int(action_probs.argmax(dim=1).item())
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = int(dist.sample().item())
            
            confidence = float(action_probs[0, action].item())
        
        return action, confidence, action_probs.cpu().numpy()[0]
    
    def predict(
        self,
        features: np.ndarray
    ) -> Tuple[int, float, np.ndarray]:
        """
        Predict action from features
        Returns: (action, confidence, q_values)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            # Ensure correct shape (batch, seq, features)
            if len(features.shape) == 2:
                features = features[np.newaxis, :]
            
            state = torch.FloatTensor(features).to(self.device)
            q_values = self.model(state)
            
            action = q_values.argmax(dim=1).item()
            probs = F.softmax(q_values, dim=1)
            confidence = probs[0, action].item()
            
        return action, confidence, q_values.cpu().numpy()[0]
    
    async def quick_train(
        self,
        df: pd.DataFrame,
        epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Quick training for cycle retrains (status-based adjustments).
        Uses reduced epochs and skips some optimizations for speed.
        Target: <2 min per cycle retrain.
        """
        logger.info(f"Quick train: {epochs} epochs for cycle retrain")
        
        # Temporarily increase epsilon for more exploration
        original_epsilon = self.epsilon
        self.epsilon = max(0.1, self.epsilon)  # At least 10% exploration
        
        # Reduce patience for quick training
        original_patience = self.early_stop_patience
        self.early_stop_patience = 3  # Quick stop if no improvement
        
        try:
            results = await self.train(df, epochs=epochs)
            results["mode"] = "quick_train"
        finally:
            # Restore original settings
            self.early_stop_patience = original_patience
        
        return results
    
    def profile_training(
        self,
        df: pd.DataFrame,
        num_batches: int = 10,
        output_path: str = "profiling/trace.json"
    ) -> Dict[str, Any]:
        """
        Profile training to identify bottlenecks.
        Runs a small number of batches and generates a Chrome trace.
        View with chrome://tracing
        """
        logger.info(f"Profiling training with {num_batches} batches")
        
        # Prepare features
        features, feature_cols = self.prepare_features(df)
        prices = df["close"].values
        
        # Initialize model if needed
        if self.model is None:
            self.initialize_model(len(feature_cols))
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        batch_size = self.ml_config.get("batch_size", 64)
        timings = {"forward": [], "backward": [], "total": []}
        
        # Determine profiler activities based on device
        activities = [ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        
        # Profile training batches
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            X, current_prices, future_prices, _ = self.create_sequences(features, prices)
            
            # Type guards
            if self.model is None:
                raise RuntimeError("model is not initialized")
            if self.replay_buffer is None:
                self.replay_buffer = ReplayBuffer(capacity=self.ml_config.get("memory_size", 10000))
            
            for i in range(min(num_batches, len(X) - 1)):
                batch_start = time.time()
                
                state = X[i:i+1]
                next_state = X[i+1:i+2]
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                with record_function("forward_pass"):
                    forward_start = time.time()
                    action, _ = self.model.get_action(state_tensor, self.epsilon)
                    forward_time = time.time() - forward_start
                    timings["forward"].append(forward_time)
                
                # Simulate reward calculation and buffer push
                price_change = (future_prices[i] - current_prices[i]) / current_prices[i]
                reward = self.calculate_reward(action, price_change)
                done = (i == num_batches - 1)
                self.replay_buffer.push(state[0], action, reward, next_state[0], done)
                
                # Train batch if enough samples
                if len(self.replay_buffer) >= batch_size:
                    with record_function("backward_pass"):
                        backward_start = time.time()
                        self._train_batch(batch_size)
                        backward_time = time.time() - backward_start
                        timings["backward"].append(backward_time)
                
                timings["total"].append(time.time() - batch_start)
        
        # Export trace
        prof.export_chrome_trace(output_path)
        logger.info(f"Profile trace saved to {output_path}")
        
        # Calculate statistics
        results = {
            "num_batches": num_batches,
            "avg_forward_ms": np.mean(timings["forward"]) * 1000 if timings["forward"] else 0,
            "avg_backward_ms": np.mean(timings["backward"]) * 1000 if timings["backward"] else 0,
            "avg_total_ms": np.mean(timings["total"]) * 1000 if timings["total"] else 0,
            "trace_path": output_path,
            "device": str(self.device)
        }
        
        logger.info(
            f"Profiling results: "
            f"Forward: {results['avg_forward_ms']:.2f}ms, "
            f"Backward: {results['avg_backward_ms']:.2f}ms, "
            f"Total: {results['avg_total_ms']:.2f}ms"
        )
        
        return results
    
    def get_signal(
        self,
        features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Get trading signal from features
        Returns: (signal_name, signal_strength, details)
        """
        action, confidence, q_values = self.predict(features)
        
        # Calculate signal strength based on Q-value margins
        sorted_q = np.sort(q_values)[::-1]
        margin = sorted_q[0] - sorted_q[1]
        
        # Normalize margin to signal strength (-1 to 1 scale)
        signal_strength = np.tanh(margin) * (1 if action == self.BUY else -1 if action == self.SELL else 0)
        
        details = {
            "q_hold": float(q_values[0]),
            "q_buy": float(q_values[1]),
            "q_sell": float(q_values[2]),
            "confidence": confidence,
            "margin": float(margin)
        }
        
        return self.ACTION_NAMES[action], signal_strength, details
    
    def save_model(self, filename: str) -> None:
        """Save model checkpoint"""
        if self.model is None:
            return
        
        filepath = self.model_dir / filename
        
        checkpoint = {
            "model_state": self.model.state_dict(),
            "target_state": self.target_model.state_dict() if self.target_model else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "epsilon": self.epsilon,
            "input_size": self.input_size,
            "feature_columns": self.feature_columns,
            "config": self.ml_config,
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str) -> bool:
        """Load model checkpoint"""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Initialize model with saved config
            self.input_size = checkpoint["input_size"]
            self.feature_columns = checkpoint.get("feature_columns", [])
            # Reset epsilon to allow exploration (don't use saved low epsilon)
            saved_epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.epsilon = max(saved_epsilon, 0.3)  # Minimum 30% exploration on reload
            
            self.initialize_model(self.input_size)
            
            # Type guards after initialization
            if self.model is None:
                raise RuntimeError("model failed to initialize")
                
            self.model.load_state_dict(checkpoint["model_state"])
            
            if checkpoint.get("target_state") and self.target_model is not None:
                self.target_model.load_state_dict(checkpoint["target_state"])
            
            if checkpoint.get("optimizer_state") and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        if self.model is None:
            return {"status": "not_initialized"}
        
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "initialized",
            "device": str(self.device),
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
            "parameter_count": param_count,
            "trainable_parameters": trainable_count,
            "epsilon": self.epsilon,
            "feature_columns": self.feature_columns,
            "replay_buffer_size": len(self.replay_buffer) if self.replay_buffer else 0
        }


# ==================== Backtesting Support ====================

class BacktestEnvironment:
    """
    Environment for backtesting ML model
    Simulates trading with realistic execution
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1000.0,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0001,
        take_profit_pct: float = 0.5,
        stop_loss_pct: float = 0.3
    ):
        self.df = df
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        
        # State
        self.capital = initial_capital
        self.position = 0.0  # Current position size
        self.position_side = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.current_idx = 0
        
        # History
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
    
    def reset(self) -> None:
        """Reset environment to initial state"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.current_idx = 0
        self.trades = []
        self.equity_curve = []
    
    def _close_position(self, current_price: float, reason: str = "signal") -> float:
        """Close current position and return PnL"""
        if self.position == 0:
            return 0.0
        
        if self.position_side == 1:  # Long
            proceeds = self.position * current_price * (1 - self.transaction_cost - self.slippage)
            pnl = proceeds - (self.position * self.entry_price)
        else:  # Short
            # For short: profit when price goes down
            cost_to_close = self.position * current_price * (1 + self.transaction_cost + self.slippage)
            proceeds_from_short = self.position * self.entry_price
            pnl = proceeds_from_short - cost_to_close
            proceeds = self.capital  # Short doesn't add proceeds directly
        
        self.capital += proceeds if self.position_side == 1 else pnl
        
        self.trades.append({
            "entry_idx": self.current_idx - 1,
            "exit_idx": self.current_idx,
            "entry_price": self.entry_price,
            "exit_price": current_price,
            "size": self.position,
            "side": "long" if self.position_side == 1 else "short",
            "pnl": pnl,
            "pnl_pct": pnl / (self.position * self.entry_price) * 100,
            "reason": reason
        })
        
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        
        return pnl / self.initial_capital
    
    def step(
        self,
        action: int,
        position_size_pct: float = 0.1
    ) -> Tuple[float, bool]:
        """
        Execute one step in the environment
        Includes automatic TP/SL exits
        Returns: (reward, done)
        """
        if self.current_idx >= len(self.df) - 1:
            return 0.0, True
        
        current_price = self.df.iloc[self.current_idx]["close"]
        next_price = self.df.iloc[self.current_idx + 1]["close"]
        
        reward = 0.0
        
        # Check TP/SL first if we have a position
        if self.position > 0 and self.entry_price > 0:
            if self.position_side == 1:  # Long position
                pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:  # Short position
                pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
            
            # Take profit
            if pnl_pct >= self.take_profit_pct:
                reward = self._close_position(current_price, "take_profit")
            # Stop loss
            elif pnl_pct <= -self.stop_loss_pct:
                reward = self._close_position(current_price, "stop_loss")
        
        # Process new signals only if flat
        if self.position == 0:
            if action == 1:  # BUY - Open long
                size = (self.capital * position_size_pct) / current_price
                cost = size * current_price * (1 + self.transaction_cost + self.slippage)
                
                if cost <= self.capital:
                    self.position = size
                    self.position_side = 1
                    self.entry_price = current_price
                    self.capital -= cost
                    
            elif action == 2:  # SELL - Open short
                size = (self.capital * position_size_pct) / current_price
                margin = size * current_price * (1 + self.transaction_cost + self.slippage)
                
                if margin <= self.capital:
                    self.position = size
                    self.position_side = -1
                    self.entry_price = current_price
                    # For short, we receive funds upfront (simplified)
        
        # Calculate equity for curve
        if self.position > 0:
            if self.position_side == 1:
                unrealized = self.position * (next_price - self.entry_price)
            else:
                unrealized = self.position * (self.entry_price - next_price)
            total_equity = self.capital + self.position * self.entry_price + unrealized
        else:
            total_equity = self.capital
        
        self.equity_curve.append(total_equity)
        self.current_idx += 1
        
        done = self.current_idx >= len(self.df) - 1
        
        return reward, done
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate backtest metrics"""
        if not self.equity_curve:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio (annualized)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(525600)
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Win rate and profit factor
        if self.trades:
            wins = [t for t in self.trades if t["pnl"] > 0]
            losses = [t for t in self.trades if t["pnl"] <= 0]
            win_rate = len(wins) / len(self.trades) * 100
            
            gross_profit = sum(t["pnl"] for t in wins)
            gross_loss = abs(sum(t["pnl"] for t in losses))
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = 2.0 if gross_profit > 0 else 1.0  # Good or neutral
        else:
            win_rate = 0.0
            profit_factor = 1.0  # Neutral when no trades
        
        # Step 1: Calculate actual backtest duration in days for proper monthly extrapolation
        if len(self.df) > 1 and "timestamp" in self.df.columns:
            try:
                start_ts = pd.to_datetime(self.df.iloc[0]["timestamp"])
                end_ts = pd.to_datetime(self.df.iloc[-1]["timestamp"])
                backtest_days = max(1, (end_ts - start_ts).total_seconds() / 86400)
            except:
                backtest_days = len(self.df) / 1440  # Assume 1m candles
        else:
            backtest_days = len(self.df) / 1440  # Assume 1m candles
        
        return {
            "total_return_pct": total_return,
            "final_equity": equity[-1],
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": np.mean([t["pnl"] for t in self.trades]) if self.trades else 0,
            "backtest_days": backtest_days  # Step 1: Include for monthly extrapolation
        }
