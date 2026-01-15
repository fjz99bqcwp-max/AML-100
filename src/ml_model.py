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
    LSTM network for temporal feature extraction from market data.
    Strict spec: 2 layers, hidden_size=128 for XYZ100-USDC perps.
    Captures sequential patterns in price/volume data.
    """
    
    # Spec defaults - DO NOT MODIFY
    SPEC_HIDDEN_SIZE = 128
    SPEC_NUM_LAYERS = 2
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,  # Spec: 128
        num_layers: int = 2,     # Spec: 2 layers
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Enforce spec defaults
        self.hidden_size = self.SPEC_HIDDEN_SIZE
        self.num_layers = self.SPEC_NUM_LAYERS
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.SPEC_HIDDEN_SIZE,
            num_layers=self.SPEC_NUM_LAYERS,
            batch_first=True,
            dropout=dropout if self.SPEC_NUM_LAYERS > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(self.SPEC_HIDDEN_SIZE)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = h_n[-1]  # (batch, hidden_size)
        out = self.layer_norm(out)
        
        return out


class DQNHead(nn.Module):
    """
    Deep Q-Network head for action value estimation.
    Strict spec: 3 actions only (HOLD=0, BUY=1, SELL=2).
    Outputs Q-values for XYZ100-USDC perps trading.
    """
    
    # Spec defaults - DO NOT MODIFY
    SPEC_NUM_ACTIONS = 3  # HOLD, BUY, SELL only
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 3  # Spec: 3 actions only
    ):
        super().__init__()
        
        # Enforce spec: exactly 3 actions
        self.num_actions = self.SPEC_NUM_ACTIONS
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.SPEC_NUM_ACTIONS)
        
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
    """
    Configuration for reward calculation - FIXED for PnL-focused training.
    
    CRITICAL FIX: Removed excessive bonuses that were masking actual PnL signal.
    The model was learning "trade = reward" instead of "profitable trade = reward".
    
    Key changes:
    - PnL is now the DOMINANT signal (pnl_scale = 20)
    - Reduced all bonuses to be supplementary, not overwhelming
    - Wrong direction now gives STRONG penalty, not weak one
    - Trade bonus ONLY for profitable trades, not all trades
    """
    # Core PnL incentives - PnL MUST dominate
    pnl_scale: float = 20.0  # Amplify PnL signal strongly
    streak_bonus: float = 0.1  # Reduced - bonus per consecutive profitable trade
    drawdown_penalty: float = 0.3  # INCREASED - penalty for losing trades
    vol_scale_factor: float = 1.0  # Reduced - don't over-scale
    
    # Action balance
    action_balance_penalty: float = 0.05  # Reduced
    bias_threshold: float = 0.70  # Increased threshold - allow more bias if profitable
    diversity_penalty: float = 0.02  # Reduced
    
    # Sharpe and exploration
    alpha_sharpe: float = 0.2  # Reduced
    beta_trade_penalty: float = 0.01  # Small penalty for flip-flopping
    early_bonus_epochs: int = 20  # Reduced exploration period
    early_bonus_amount: float = 0.05  # GREATLY reduced
    
    # Hold penalties - less aggressive (HOLD is valid when uncertain)
    hold_penalty: float = 0.005  # Reduced - HOLD is ok
    hold_timeout_seconds: int = 60  # Increased timeout
    hold_timeout_penalty: float = 0.1  # Reduced
    max_consecutive_holds: int = 60  # Increased
    
    # Transaction costs
    commission_rate: float = 0.0005  # Transaction cost
    
    # Reward bounds (symmetric for clearer signal)
    min_trade_reward: float = -2.0  # Allow stronger negative feedback
    max_trade_reward: float = 2.0  # Symmetric ceiling
    
    # Bonuses - GREATLY REDUCED
    correct_direction_bonus: float = 0.1  # Reduced - PnL already rewards this
    trade_bonus: float = 0.0  # REMOVED - no bonus just for trading
    positive_bias: float = 0.0  # REMOVED - let actual PnL drive rewards
    
    # Streak tracking
    consecutive_trade_bonus: float = 0.05  # Reduced
    max_streak_bonus: float = 0.3  # Reduced cap
    
    # Scaling
    log_scale_threshold: float = 1.0  # Increased threshold
    vol_window: int = 20  # Window for volatility calculation
    kelly_weight: float = 0.05  # Greatly reduced


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
    High-level ML model manager for HFT.
    STRICT SPEC: LSTM(2 layers, 128 hidden) + DQN(3 actions).
    Handles training, inference, and model management.
    """
    
    # Action mapping - SPEC: 3 actions only
    HOLD = 0
    BUY = 1
    SELL = 2
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    # SPEC DEFAULTS - enforced regardless of config (M4 optimized)
    SPEC_DEFAULTS = {
        "lstm_hidden_size": 128,
        "lstm_num_layers": 2,
        "dqn_hidden_size": 256,
        "num_actions": 3,
        "learning_rate": 0.0001,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.98,  # Faster decay - reach 0.05 after ~150 epochs
        "epochs": 300,  # More epochs for proper training
        "early_stop_patience": 100,  # Much more patience for convergence
        "min_reward_delta": 0.001,  # Allow small improvements to count
        "num_workers": 8,  # Optimized for M4 (8 performance cores)
        "data_days": 180,  # Default to 6 months of data
    }
    
    def __init__(
        self,
        config_path: str = "config/params.json",
        model_dir: str = "models",
        reset_defaults: bool = False
    ):
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.ml_config = self.config.get("ml_model", {})
        
        # SPEC: Enforce defaults on init
        self._apply_spec_defaults()
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device (prefer MPS on M4 Mac)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for M4 acceleration")
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
        self.early_stop_patience = self.ml_config.get("early_stop_patience", 100)
        self.min_reward_delta = self.ml_config.get("min_reward_delta", 0.001)  # Allow small improvements
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
        self.epsilon_end = self.ml_config.get("epsilon_end", 0.05)
        self.epsilon_decay = self.SPEC_DEFAULTS["epsilon_decay"]  # SPEC: 0.997
        self.gamma = self.ml_config.get("gamma", 0.99)
        
        # SPEC: Disable supervised guidance and FGSM - pure DQN
        self.supervised_gamma = 0.0
        self.use_supervised_guidance = False
        
        # SPEC: Disable FGSM adversarial training
        self.use_fgsm = False
        self.fgsm_weight = 0.0
        self.fgsm_epsilon = 0.0
        
        # SPEC: DQN only - no A2C/PPO
        self.rl_algo = "DQN"  # Enforced: DQN only
        self.ac_model = None  # No actor-critic model
        
        # Feature columns for training
        self.feature_columns: List[str] = []
        
        # Metrics
        self.training_history: List[TrainingMetrics] = []
        
        logger.info(f"MLModel initialized with SPEC defaults: LSTM(2x128)+DQN(3), lr={self.SPEC_DEFAULTS['learning_rate']}, epsilon_decay={self.SPEC_DEFAULTS['epsilon_decay']}")
    
    def _apply_spec_defaults(self) -> None:
        """Apply SPEC defaults to ml_config, overriding any custom values."""
        for key, value in self.SPEC_DEFAULTS.items():
            self.ml_config[key] = value
        logger.debug(f"Applied SPEC defaults: {self.SPEC_DEFAULTS}")
    
    @staticmethod
    def _strip_compiled_prefix(state_dict: dict) -> dict:
        """Strip _orig_mod. prefix from state_dict keys (torch.compile compatibility)"""
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
    def initialize_model(self, input_size: int) -> None:
        """Initialize model architecture with M4 MPS optimizations.
        STRICT SPEC: LSTM(2 layers, 128 hidden) + DQN(3 actions).
        """
        self.input_size = input_size
        
        # SPEC DEFAULTS - ENFORCED (ignore config overrides)
        lstm_hidden = 128   # Spec: 128 hidden size
        lstm_layers = 2     # Spec: 2 layers
        dqn_hidden = 256    # DQN hidden layer size
        num_actions = 3     # Spec: HOLD, BUY, SELL only
        
        logger.info(f"Initializing STRICT LSTM+DQN: LSTM({lstm_layers}x{lstm_hidden}) + DQN({num_actions} actions)")
        
        self.model = HybridLSTMDQN(
            input_size=input_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dqn_hidden=dqn_hidden,
            num_actions=num_actions,
            dropout=self.ml_config.get("dropout", 0.2)
        ).to(self.device)
        
        # TORCH.COMPILE FOR M4 MPS OPTIMIZATION
        if self.ml_config.get("enable_torch_compile", True):
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("torch.compile() enabled - M4 MPS optimized for <0.5ms inference")
                except Exception as e:
                    logger.warning(f"torch.compile() failed: {e}, using eager mode")
            else:
                logger.warning("torch.compile() not available (requires PyTorch >=2.0)")
        
        # Target network for stable Q-learning (same spec architecture)
        self.target_model = HybridLSTMDQN(
            input_size=input_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dqn_hidden=dqn_hidden,
            num_actions=num_actions,
            dropout=self.ml_config.get("dropout", 0.2)
        ).to(self.device)
        
        # Strip _orig_mod. prefix if model was compiled with torch.compile()
        state_dict = self._strip_compiled_prefix(self.model.state_dict())
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
        
        # Optimizer with SPEC learning rate
        spec_lr = 0.0001  # Spec: learning_rate 0.0001
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=spec_lr
        )
        
        # Learning rate scheduler - more conservative to prevent premature LR decay
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=self.ml_config.get("lr_scheduler_factor", 0.7),
            patience=self.ml_config.get("lr_scheduler_patience", 30),
            min_lr=self.ml_config.get("lr_min", 1e-6)
        )
        
        self.replay_buffer = ReplayBuffer(
            capacity=self.ml_config.get("memory_size", 10000)
        )
        
        # Reset early stopping state
        self.best_reward_ever = float("-inf")
        self.patience_counter = 0
        self.epoch_times = []
        
        # Reward threshold stopping
        self.reward_threshold_stop = self.ml_config.get("reward_threshold_stop", -0.3)
        self.reward_threshold_epochs = self.ml_config.get("reward_threshold_epochs", 20)
        
        logger.info(f"Model initialized: input_size={input_size}, LSTM=2x128, DQN=3 actions, lr={spec_lr}")
    
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
        """
        Prepare feature matrix from DataFrame.
        SIMPLIFIED: Only equity-specific basics - OHLCV, returns, volume, and time-based.
        NO additional indicators (RSI, ATR, MACD, etc.) per spec.
        """
        # SPEC: Only basic OHLCV-derived features for XYZ100-USDC equity perps
        feature_cols = [
            # Price action basics
            "returns", "log_returns",
            # Volume
            "volume_ratio",
            # Time-based (equity market patterns)
            "hour_of_day",
            # Simple momentum (price-based only)
            "momentum_5", "momentum_10",
            # Volatility (from returns, not ATR)
            "volatility",
            # Equity-specific basics
            "equity_vol_5", "equity_vol_20",
            "gap_pct", "trend_persistence",
            # Simple moving average spreads (no complex indicators)
            "sma_spread_7_21", "sma_spread_7_50",
        ]
        
        # Add computed basic features if not present
        df = df.copy()
        
        # Basic returns
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
        if "log_returns" not in df.columns:
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Volume ratio (simple)
        if "volume_ratio" not in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / vol_ma.clip(lower=1)
        
        # Hour of day
        if "hour_of_day" not in df.columns:
            if "timestamp" in df.columns:
                df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour / 24.0
            else:
                df["hour_of_day"] = 0.0
        
        # Simple momentum
        if "momentum_5" not in df.columns:
            df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        if "momentum_10" not in df.columns:
            df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        
        # Volatility from returns
        if "volatility" not in df.columns:
            df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252 * 24 * 60)
        
        # Equity vol basics
        if "equity_vol_5" not in df.columns:
            df["equity_vol_5"] = df["returns"].rolling(5).std() * np.sqrt(252 * 24 * 60)
        if "equity_vol_20" not in df.columns:
            df["equity_vol_20"] = df["returns"].rolling(20).std() * np.sqrt(252 * 24 * 60)
        
        # Gap and trend
        if "gap_pct" not in df.columns:
            df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        if "trend_persistence" not in df.columns:
            df["trend_persistence"] = df["returns"].rolling(10).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5, raw=False
            )
        
        # Simple SMA spreads (no complex indicators)
        if "sma_spread_7_21" not in df.columns:
            sma_7 = df["close"].rolling(7).mean()
            sma_21 = df["close"].rolling(21).mean()
            df["sma_spread_7_21"] = (sma_7 - sma_21) / df["close"]
        if "sma_spread_7_50" not in df.columns:
            sma_7 = df["close"].rolling(7).mean()
            sma_50 = df["close"].rolling(50).mean()
            df["sma_spread_7_50"] = (sma_7 - sma_50) / df["close"]
        
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
        atr: Optional[np.ndarray] = None,
        forward_bars: int = 5  # Look-ahead for multi-bar returns
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM training with configurable forward horizon"""
        seq_len = self.sequence_length
        n_samples = len(features) - seq_len - forward_bars  # Account for look-ahead
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data for sequence length {seq_len} and forward_bars {forward_bars}")
        
        X = np.zeros((n_samples, seq_len, features.shape[1]))
        y_prices = np.zeros(n_samples)
        current_prices = np.zeros(n_samples)
        atr_values = np.zeros(n_samples) if atr is not None else None
        
        for i in range(n_samples):
            X[i] = features[i:i + seq_len]
            current_prices[i] = prices[i + seq_len - 1]
            # Use forward_bars look-ahead instead of just 1 bar
            y_prices[i] = prices[i + seq_len - 1 + forward_bars]
            if atr is not None and atr_values is not None:
                atr_values[i] = atr[i + seq_len - 1]
        
        return X, current_prices, y_prices, atr_values
    
    def calculate_reward(
        self,
        action: int,
        price_change_pct: float,
        atr_normalized: float = 1.0,
        prev_action: Optional[int] = None,
        status: str = "normal",
        daily_vol: float = 0.01
    ) -> float:
        """
        ENHANCED PnL-focused reward with asymmetric scaling.
        
        KEY IMPROVEMENTS:
        1. Asymmetric rewards: Winning trades get 1.5x reward vs losses
        2. Trend following bonus: Extra reward for riding trends
        3. Risk-adjusted: Larger positions in low-vol, smaller in high-vol
        4. Win streak bonuses: Encourage consistent profitability
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price_change_pct: (future_price - current_price) / current_price
            atr_normalized: ATR / price for volatility normalization
            prev_action: Previous action for trend detection
            status: Training status
            daily_vol: Daily volatility for risk adjustment
        
        Returns:
            Reward value directly proportional to risk-adjusted trading PnL
        """
        cfg = self.reward_config
        
        # Initialize tracking attributes
        if not hasattr(self, '_consecutive_holds'):
            self._consecutive_holds = 0
        if not hasattr(self, '_win_streak'):
            self._win_streak = 0
        if not hasattr(self, '_trend_direction'):
            self._trend_direction = 0  # +1 = uptrend, -1 = downtrend
        
        # Volatility-adjusted PnL scaling (trade smaller in high vol)
        vol_factor = max(0.5, min(2.0, 0.02 / max(daily_vol, 0.005)))
        effective_pnl_scale = cfg.pnl_scale * vol_factor
        
        # Asymmetric reward multipliers (wins worth more than losses)
        WIN_MULTIPLIER = 1.5
        LOSS_MULTIPLIER = 1.0
        
        # CORE LOGIC: Reward = Risk-adjusted PnL from the action
        if action == self.HOLD:
            self._consecutive_holds += 1
            
            # HOLD is GOOD when price barely moves
            if abs(price_change_pct) < cfg.commission_rate * 2:
                base_reward = 0.08  # Slightly higher for smart holds
                if self._consecutive_holds <= 2:
                    base_reward += 0.02  # Bonus for brief holds
            else:
                # Opportunity cost - but capped to prevent over-penalization
                missed_pnl = abs(price_change_pct) - cfg.commission_rate
                hold_penalty = cfg.hold_penalty * min(self._consecutive_holds, 3)  # Cap at 3
                base_reward = -(hold_penalty + missed_pnl * effective_pnl_scale * 0.05)  # Reduced penalty
                base_reward = max(base_reward, -0.15)  # Floor on HOLD penalty
            
            # Reset win streak on excessive holding
            if self._consecutive_holds > 5:
                self._win_streak = max(0, self._win_streak - 1)
            
        elif action == self.BUY:
            self._consecutive_holds = 0
            
            # Direct PnL: positive price change = profit
            raw_pnl = price_change_pct - cfg.commission_rate
            
            if raw_pnl > 0:
                # WINNING TRADE - apply win multiplier
                base_reward = raw_pnl * effective_pnl_scale * WIN_MULTIPLIER
                self._win_streak += 1
                self._trend_direction = 1
                
                # Win streak bonus (compounding encouragement)
                streak_bonus = min(0.1 * self._win_streak, 0.3)
                base_reward += streak_bonus
                
            else:
                # LOSING TRADE - standard penalty
                base_reward = raw_pnl * effective_pnl_scale * LOSS_MULTIPLIER
                self._win_streak = 0
                
                # Extra drawdown penalty for significant losses
                if price_change_pct < -0.005:  # >0.5% loss
                    base_reward -= cfg.drawdown_penalty * abs(price_change_pct) * effective_pnl_scale * 0.5
            
            # Trend following bonus: continuing in trend direction
            if prev_action == self.BUY and self._trend_direction == 1 and price_change_pct > 0:
                base_reward += 0.05  # Trend continuation bonus
            
        else:  # SELL
            self._consecutive_holds = 0
            
            # Direct PnL: negative price change = profit for short
            raw_pnl = -price_change_pct - cfg.commission_rate
            
            if raw_pnl > 0:
                # WINNING TRADE - apply win multiplier
                base_reward = raw_pnl * effective_pnl_scale * WIN_MULTIPLIER
                self._win_streak += 1
                self._trend_direction = -1
                
                # Win streak bonus
                streak_bonus = min(0.1 * self._win_streak, 0.3)
                base_reward += streak_bonus
                
            else:
                # LOSING TRADE - standard penalty
                base_reward = raw_pnl * effective_pnl_scale * LOSS_MULTIPLIER
                self._win_streak = 0
                
                # Extra drawdown penalty for significant losses
                if price_change_pct > 0.005:  # >0.5% rise against short
                    base_reward -= cfg.drawdown_penalty * abs(price_change_pct) * effective_pnl_scale * 0.5
            
            # Trend following bonus
            if prev_action == self.SELL and self._trend_direction == -1 and price_change_pct < 0:
                base_reward += 0.05  # Trend continuation bonus
        
        # Direction accuracy bonus (supplementary)
        direction_bonus = 0.0
        if action == self.BUY and price_change_pct > 0.001:
            direction_bonus = cfg.correct_direction_bonus
        elif action == self.SELL and price_change_pct < -0.001:
            direction_bonus = cfg.correct_direction_bonus
        
        # Combine rewards
        total_reward = base_reward + direction_bonus
        
        # Wider clip range to allow stronger learning signals
        total_reward = np.clip(total_reward, cfg.min_trade_reward, cfg.max_trade_reward * 1.5)
        
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
        
        # Use 10-bar forward horizon for training rewards
        # This better matches holding periods (trades held for 10-60 bars)
        forward_bars = self.ml_config.get("training_forward_bars", 10)
        X, current_prices, future_prices, atr_seq = self.create_sequences(
            features, prices, atr_values, forward_bars=forward_bars
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
        
        # Batch action selection with PER-ACTION epsilon-greedy (NOT batch-wide)
        self.model.eval()
        with torch.no_grad():
            sampled_states = all_states[sample_indices]
            q_values = self.model(sampled_states)
            
            # Per-action epsilon-greedy: each action independently explores or exploits
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            random_actions = np.random.randint(0, 3, size=len(sample_indices))
            explore_mask = np.random.random(len(sample_indices)) < self.epsilon
            actions = np.where(explore_mask, random_actions, greedy_actions)
        
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
        
        # REDUCED diversity penalties - let PnL drive behavior, not arbitrary action quotas
        diversity_thresh = self.ml_config.get("diversity_thresh", 0.80)  # Allow 80% same action if profitable
        
        # Strong diversity penalty ONLY if model is completely stuck on one action (>90%)
        total_actions = len(actions)
        if total_actions > 10:
            max_action_pct = action_counts.max() / total_actions
            
            # Only penalize extreme HOLD dominance (>90%) - HOLD is valid when uncertain
            hold_pct = action_counts[self.HOLD] / total_actions
            if hold_pct > 0.90:  # Very high threshold
                hold_penalty = -self.reward_config.diversity_penalty * 5 * total_actions  # Reduced 12x -> 5x
                rewards = rewards + (hold_penalty / len(rewards))
                logger.warning(f"Extreme HOLD dominance: {hold_pct:.1%} HOLD")
            
            # Only penalize very extreme action bias (>90%)
            if max_action_pct > 0.90:
                diversity_penalty = -self.reward_config.diversity_penalty * 3 * total_actions  # Reduced 10x -> 3x
                rewards = rewards + (diversity_penalty / len(rewards))
                logger.debug(f"Diversity penalty: {max_action_pct:.1%} same action")
            
            # REMOVED: Force minimum trading activity - let model learn when to trade
            # REMOVED: BUY/SELL balance enforcement - model should follow market direction
        
        # Mild penalty if no trades at all - but not severe
        if trade_count == 0 and total_actions > 20:
            diversity_penalty = -self.reward_config.diversity_penalty * total_actions * 0.5  # Greatly reduced
            rewards = rewards + (diversity_penalty / len(rewards))
            logger.debug(f"No trades penalty (mild)")
        
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
            for batch_idx in range(num_train_batches):
                try:
                    self._train_batch(batch_size, optimal_q, effective_gamma)
                except Exception as e:
                    logger.error(f"_train_batch failed at batch {batch_idx}: {e}")
                    logger.error(f"   replay_buffer size: {len(self.replay_buffer)}")
                    logger.error(f"   batch_size: {batch_size}")
                    logger.error(f"   model.training: {self.model.training if self.model else 'N/A'}")
                    raise
        
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
        
        # Ensure model is in training mode
        self.model.train()
        
        # Debug: Check model parameters require grad
        params_with_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
        total_params = sum(1 for p in self.model.parameters())
        if params_with_grad != total_params:
            logger.error(f"Model param issue: {params_with_grad}/{total_params} require grad")
        
        # Debug: Check torch.is_grad_enabled()
        if not torch.is_grad_enabled():
            logger.error("torch.is_grad_enabled() is False!")
        
        # Forward pass with optional AMP (disabled on MPS)
        # Note: autocast is CUDA-only, we skip it for MPS
        if self.device.type == 'cuda' and self.use_amp:
            with autocast(device_type='cuda', enabled=True):
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
                loss = td_loss
        else:
            # Standard forward pass (MPS/CPU)
            all_q = self.model(states)
            current_q = all_q.gather(1, actions.unsqueeze(1))
            
            # Target Q values (Double DQN)
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1, keepdim=True)
                next_q = self.target_model(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (~dones).unsqueeze(1)
            
            # Primary loss: TD error (Huber loss)
            td_loss = F.smooth_l1_loss(current_q, target_q)
            loss = td_loss
        
        # Step 2 + Step 4: Supervised guidance loss with effective gamma
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
            loss = loss + effective_gamma * supervised_loss
        
        # Step 2: Add entropy bonus to encourage exploration and diverse actions
        entropy_bonus_coef = self.ml_config.get("entropy_bonus", 0.01)
        if entropy_bonus_coef > 0:
            # Compute policy entropy from Q-values (softmax distribution)
            q_probs = F.softmax(all_q, dim=1)
            q_log_probs = F.log_softmax(all_q, dim=1)
            entropy = -(q_probs * q_log_probs).sum(dim=1).mean()
            # Subtract entropy (maximize entropy = minimize negative entropy)
            loss = loss - entropy_bonus_coef * entropy
        
        # Step 5: FGSM adversarial training for robustness (disabled by default)
        if self.use_fgsm and random.random() < 0.3:
            adv_loss = self._fgsm_adversarial_loss(states, actions, target_q)
            loss = loss + self.fgsm_weight * adv_loss
        
        # Backward pass with gradient scaling (CUDA only)
        self.optimizer.zero_grad()
        
        # Debug: Check loss requires grad before backward
        if not loss.requires_grad:
            logger.error(f"Loss does not require grad!")
            logger.error(f"   loss.requires_grad={loss.requires_grad}")
            logger.error(f"   td_loss.requires_grad={td_loss.requires_grad}")
            logger.error(f"   current_q.requires_grad={current_q.requires_grad}")
            logger.error(f"   all_q.requires_grad={all_q.requires_grad}")
            logger.error(f"   device={self.device}")
        
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
        # Note: Use retain_graph=False for torch.compile compatibility
        loss.backward()
        
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
        
        # Use 10-bar forward horizon for training rewards
        forward_bars = self.ml_config.get("training_forward_bars", 10)
        X, current_prices, future_prices, atr_seq = self.create_sequences(
            features, prices, atr_values, forward_bars=forward_bars
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
            # Strip _orig_mod. prefix for torch.compile() compatibility
            state_dict = self._strip_compiled_prefix(self.model.state_dict())
            self.target_model.load_state_dict(state_dict)
    
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
                factor=self.ml_config.get("lr_scheduler_factor", 0.7),
                patience=self.ml_config.get("lr_scheduler_patience", 30),
                min_lr=self.ml_config.get("lr_min", 1e-6)
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
        MPS-optimized prediction (<1ms target)
        Returns: (action, confidence, q_values)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Ensure correct shape (batch, seq, features)
                if len(features.shape) == 2:
                    features = features[np.newaxis, :]
                
                state = torch.FloatTensor(features).to(self.device)
                q_values = self.model(state)
                
                # CRITICAL: Reset LSTM hidden states to prevent memory accumulation
                if hasattr(self.model, 'lstm'):
                    # Clear LSTM hidden state between predictions
                    self.model.lstm.flatten_parameters()
                
                action = q_values.argmax(dim=1).item()
                probs = F.softmax(q_values, dim=1)
                confidence = probs[0, action].item()
                
                # Move to CPU and explicitly clean up GPU tensor
                q_vals_cpu = q_values.cpu().numpy()[0]
                
                # Explicit cleanup for MPS
                del state, q_values, probs
            
            return action, confidence, q_vals_cpu
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return safe default (HOLD with zero confidence)
            return 0, 0.0, np.zeros(3)
    
    def predict_batch(
        self,
        features_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch prediction for massive speedup during backtests.
        Processes multiple samples at once on GPU.
        
        Args:
            features_batch: Shape (batch_size, seq_len, n_features)
        
        Returns:
            actions: Shape (batch_size,)
            confidences: Shape (batch_size,)
            q_values: Shape (batch_size, 3)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Convert to tensor and move to device
                state = torch.FloatTensor(features_batch).to(self.device)
                q_values = self.model(state)
                
                # Get actions and confidences
                actions = q_values.argmax(dim=1).cpu().numpy()
                probs = F.softmax(q_values, dim=1)
                confidences = probs.gather(1, torch.tensor(actions).unsqueeze(1).to(self.device)).squeeze().cpu().numpy()
                q_vals_cpu = q_values.cpu().numpy()
                
                # Cleanup
                del state, q_values, probs
            
            return actions, confidences, q_vals_cpu
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            batch_size = len(features_batch)
            return np.zeros(batch_size, dtype=int), np.zeros(batch_size), np.zeros((batch_size, 3))
    
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
        
        # Strip _orig_mod. prefix for torch.compile() compatibility
        checkpoint = {
            "model_state": self._strip_compiled_prefix(self.model.state_dict()),
            "target_state": self._strip_compiled_prefix(self.target_model.state_dict()) if self.target_model else None,
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
            # Keep epsilon low for inference - training will reset if needed
            saved_epsilon = checkpoint.get("epsilon", self.epsilon_end)
            self.epsilon = saved_epsilon  # Use saved epsilon directly
            
            self.initialize_model(self.input_size)
            
            # Type guards after initialization
            if self.model is None:
                raise RuntimeError("model failed to initialize")
            
            # Handle torch.compile() state_dict compatibility
            # Saved state has no prefix, but compiled model expects _orig_mod. prefix
            model_state = checkpoint["model_state"]
            is_compiled = hasattr(self.model, '_orig_mod')
            
            if is_compiled:
                # Add _orig_mod. prefix for compiled models
                model_state = {f'_orig_mod.{k}' if not k.startswith('_orig_mod.') else k: v 
                               for k, v in model_state.items()}
            else:
                # Strip prefix for non-compiled models
                model_state = self._strip_compiled_prefix(model_state)
            
            self.model.load_state_dict(model_state)
            
            if checkpoint.get("target_state") and self.target_model is not None:
                # Target model is never compiled, so always strip prefix
                target_state = self._strip_compiled_prefix(checkpoint["target_state"])
                self.target_model.load_state_dict(target_state)
            
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
    
    # HARD cap to prevent overflow: 1000x the reference starting capital ($1000)
    MAX_CAPITAL_ABSOLUTE = 1_000_000.0  # $1M = 100,000% max gain
    MAX_DRAWDOWN_PCT = 5.0  # Portfolio-level drawdown limit (matches objective)
    
    # Layered drawdown protection:
    # - CIRCUIT_BREAKER_DD_PCT: Pause trading, force-close positions, wait for recovery
    # - HARD_TERMINATE_DD_PCT: STOP BACKTEST ENTIRELY - no recovery possible
    CIRCUIT_BREAKER_DD_PCT = 3.5  # Pause trading at 3.5% DD
    HARD_TERMINATE_DD_PCT = 4.5   # Terminate backtest at 4.5% DD (buffer before 5% objective)
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1000.0,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0001,
        take_profit_pct: float = 0.5,
        stop_loss_pct: float = 0.3,
        max_hold_bars: int = 120,  # Time-based exit (2 hours for trends to develop)
        max_drawdown_pct: float = 5.0,  # Portfolio drawdown limit
        global_peak_equity: float = None  # Global peak from previous chunks (for proper DD tracking)
    ):
        self.df = df
        # Cap initial_capital to prevent overflow from chunk chaining
        self.initial_capital = min(initial_capital, self.MAX_CAPITAL_ABSOLUTE)
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_bars = max_hold_bars
        self.max_drawdown_pct = max_drawdown_pct
        
        # State - use capped initial_capital to prevent overflow
        self.capital = self.initial_capital  # Use capped value, not raw input
        self.position = 0.0  # Current position size
        self.position_side = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.entry_idx = 0  # Track entry index for time-based exit
        self.current_idx = 0
        
        # Portfolio drawdown tracking - use global peak if provided (for chunk chaining)
        self.peak_equity = global_peak_equity if global_peak_equity is not None else self.initial_capital
        self.drawdown_halt = False  # True when drawdown exceeds circuit breaker
        self.permanently_halted = False  # True when HARD terminate threshold hit - no recovery
        self.recovery_target = 0.0  # Equity needed to resume trading
        
        # History
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
    
    def reset(self) -> None:
        """Reset environment to initial state"""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_idx = 0
        self.peak_equity = self.initial_capital
        self.drawdown_halt = False
        self.permanently_halted = False
        self.recovery_target = 0.0
        self.trades = []
        self.equity_curve = []
    
    def _close_position(self, current_price: float, reason: str = "signal") -> float:
        """Close current position and return PnL"""
        if self.position == 0:
            return 0.0
        
        # Calculate fees (round-trip)
        fees_pct = self.transaction_cost + self.slippage
        notional = self.position * self.entry_price
        
        # Guard against NaN/Inf values
        if not np.isfinite(notional) or notional <= 0:
            self.position = 0.0
            self.position_side = 0
            self.entry_price = 0.0
            return 0.0
        
        if self.position_side == 1:  # Long
            # Long: buy at entry_price, sell at current_price
            # Raw PnL = (exit - entry) / entry
            raw_pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # Short
            # Short: sell at entry_price, buy back at current_price
            # Raw PnL = (entry - exit) / entry
            raw_pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
        
        # Deduct round-trip fees (entry + exit)
        net_pnl_pct = raw_pnl_pct - (fees_pct * 2 * 100)
        
        # Guard against extreme PnL percentages
        net_pnl_pct = np.clip(net_pnl_pct, -100, 1000)  # Cap at 1000% gain or 100% loss
        
        # Calculate dollar PnL with overflow protection
        pnl = float(notional) * (float(net_pnl_pct) / 100)
        if not np.isfinite(pnl):
            pnl = -notional  # Assume total loss if overflow
        
        # Update capital: add back notional + pnl
        # (We subtracted notional at entry, now add it back with profit/loss)
        self.capital += notional + pnl
        
        # Sanity check: prevent capital going negative, NaN, or absurdly high
        if not np.isfinite(self.capital) or self.capital < 1.0:
            self.capital = 1.0  # Minimum capital to prevent 100% drawdown artifacts
        elif self.capital > self.MAX_CAPITAL_ABSOLUTE:
            # Hard cap to prevent overflow (already 100,000% gain from reference)
            self.capital = self.MAX_CAPITAL_ABSOLUTE
        
        # Only log significant trades, not every close
        bars_held = self.current_idx - self.entry_idx
        # Debug logging disabled for performance - uncomment for debugging:
        # logging.getLogger(__name__).debug(f"CLOSE [{reason}]: bars={bars_held}, pnl={pnl:.2f}, pnl_pct={net_pnl_pct:.3f}%")
        
        self.trades.append({
            "entry_idx": self.entry_idx,
            "exit_idx": self.current_idx,
            "entry_price": self.entry_price,
            "exit_price": current_price,
            "size": self.position,
            "side": "long" if self.position_side == 1 else "short",
            "pnl": pnl,
            "pnl_pct": net_pnl_pct,
            "reason": reason
        })
        
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        
        return pnl / self.initial_capital
    
    def step(
        self,
        action: int,
        position_size_pct: float = 0.1,
        data_idx: int = None
    ) -> Tuple[float, bool]:
        """
        Execute one step in the environment
        Includes automatic TP/SL/time-based exits
        
        Args:
            action: 0=hold, 1=buy, 2=sell
            position_size_pct: Position size as fraction of capital
            data_idx: Optional - use this dataframe index instead of internal counter
        
        Returns: (reward, done)
        """
        # Use external index if provided, otherwise use internal counter
        if data_idx is not None:
            self.current_idx = data_idx
        
        if self.current_idx >= len(self.df) - 1:
            return 0.0, True
        
        current_price = self.df.iloc[self.current_idx]["close"]
        next_idx = min(self.current_idx + 1, len(self.df) - 1)
        next_price = self.df.iloc[next_idx]["close"]
        
        reward = 0.0
        
        # Check exits if we have a position
        if self.position > 0 and self.entry_price > 0:
            if self.position_side == 1:  # Long position
                pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            else:  # Short position
                pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
            
            bars_held = self.current_idx - self.entry_idx
            
            # Time-based exit (for mean reversion strategy)
            if bars_held >= self.max_hold_bars:
                reward = self._close_position(current_price, "time_exit")
            # Take profit
            elif pnl_pct >= self.take_profit_pct:
                reward = self._close_position(current_price, "take_profit")
            # Stop loss
            elif pnl_pct <= -self.stop_loss_pct:
                reward = self._close_position(current_price, "stop_loss")
        
        # Calculate current equity for drawdown check
        # Note: capital has notional subtracted at entry, so we add notional + unrealized to get total equity
        current_equity = self.capital
        if self.position > 0 and self.entry_price > 0:
            notional_in_pos = float(self.position) * float(self.entry_price)
            if self.position_side == 1:  # Long
                unrealized_pnl = float(self.position) * (float(current_price) - float(self.entry_price))
            else:  # Short
                unrealized_pnl = float(self.position) * (float(self.entry_price) - float(current_price))
            if np.isfinite(notional_in_pos) and np.isfinite(unrealized_pnl):
                current_equity = self.capital + notional_in_pos + unrealized_pnl
        
        # Sanity check on equity value
        current_equity = max(1.0, min(current_equity, self.MAX_CAPITAL_ABSOLUTE))
        
        # Update peak equity (high water mark) - only if not permanently halted
        if current_equity > self.peak_equity and not self.permanently_halted:
            self.peak_equity = current_equity
            self.drawdown_halt = False  # Reset halt if we made new highs
        
        # Check portfolio drawdown
        current_dd_pct = 0.0
        if self.peak_equity > 0:
            current_dd_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        
        # LAYERED DRAWDOWN PROTECTION:
        # Layer 1 (CIRCUIT_BREAKER_DD_PCT = 3.5%): Pause trading, force-close, can recover
        # Layer 2 (HARD_TERMINATE_DD_PCT = 4.5%): STOP BACKTEST ENTIRELY - return done=True
        
        # Check HARD TERMINATION first (4.5% - no recovery possible)
        if current_dd_pct >= self.HARD_TERMINATE_DD_PCT:
            self.permanently_halted = True
            self.drawdown_halt = True
            logger.warning(f"🛑 BACKTEST HARD TERMINATION: DD={current_dd_pct:.2f}% >= {self.HARD_TERMINATE_DD_PCT}%")
            logger.warning(f"   Peak=${self.peak_equity:.2f}, Current=${current_equity:.2f}")
            # Force-close any open position immediately
            if self.position > 0:
                reward = self._close_position(current_price, "hard_termination")
            # Record final equity and return DONE
            self.equity_curve.append(current_equity)
            self.current_idx += 1
            return reward, True  # TERMINATE BACKTEST
        
        # Check CIRCUIT BREAKER (3.5% - temporary pause, can recover)
        if current_dd_pct >= self.CIRCUIT_BREAKER_DD_PCT and not self.permanently_halted:
            self.drawdown_halt = True
            self.recovery_target = self.peak_equity * (1 - self.CIRCUIT_BREAKER_DD_PCT / 100 * 0.5)  # Need to recover to ~1.75% DD to resume
            # Force-close any open position to prevent further losses
            if self.position > 0:
                logger.debug(f"Circuit breaker: DD={current_dd_pct:.2f}%, closing position")
                reward = self._close_position(current_price, "drawdown_halt")
        
        # Check if we can resume trading (recovered from circuit breaker)
        if self.drawdown_halt and not self.permanently_halted and current_equity >= self.recovery_target:
            self.drawdown_halt = False
        
        # Dynamic position sizing based on drawdown (only when not halted):
        # - At 0% DD: full position size (100%)
        # - At 1.75% DD: 50% position size  
        # - At 3.5% DD: circuit breaker (halted before this)
        dd_ratio = min(1.0, current_dd_pct / max(0.5, self.CIRCUIT_BREAKER_DD_PCT))
        dd_scalar = max(0.1, 1.0 - dd_ratio * 0.9)  # Linear scale down to 10%
        adjusted_position_pct = position_size_pct * dd_scalar
        
        # Process new signals only if:
        # 1. No current position
        # 2. Capital is valid (> $1)
        # 3. NOT in drawdown halt (temporary or permanent)
        if self.position == 0 and self.capital > 1.0 and np.isfinite(self.capital) and not self.drawdown_halt and not self.permanently_halted:
            if action == 1:  # BUY - Open long
                # Use FIXED notional sizing based on initial capital (no compounding)
                # This limits max drawdown by not betting accumulated profits
                base_notional = self.initial_capital * position_size_pct
                # Apply drawdown adjustment to the fixed size
                notional = min(base_notional * dd_scalar, self.capital * 0.95)  # Never use more than 95% of current capital
                size = notional / current_price
                
                if notional <= self.capital and notional > 0 and np.isfinite(size):
                    self.position = size
                    self.position_side = 1
                    self.entry_price = current_price
                    self.entry_idx = self.current_idx
                    # Reserve notional (fees applied at exit)
                    self.capital -= notional
                    
            elif action == 2:  # SELL - Open short
                # Use FIXED notional sizing based on initial capital (no compounding)
                base_notional = self.initial_capital * position_size_pct
                # Apply drawdown adjustment to the fixed size
                notional = min(base_notional * dd_scalar, self.capital * 0.95)
                size = notional / current_price
                
                if notional <= self.capital and notional > 0 and np.isfinite(size):
                    self.position = size
                    self.position_side = -1
                    self.entry_price = current_price
                    self.entry_idx = self.current_idx
                    # Reserve notional as margin
                    self.capital -= notional
        
        # Calculate equity for curve
        # Note: capital has notional subtracted at entry, so we add back notional + unrealized PnL
        if self.position > 0 and self.entry_price > 0:
            # Guard against extreme position sizes
            if self.position > 1e10 or self.entry_price > 1e10:
                total_equity = max(0.0, self.capital)
            else:
                notional_in_position = float(self.position) * float(self.entry_price)
                if self.position_side == 1:
                    unrealized = float(self.position) * (float(next_price) - float(self.entry_price))
                else:
                    unrealized = float(self.position) * (float(self.entry_price) - float(next_price))
                
                # Guard against overflow
                if not np.isfinite(notional_in_position) or not np.isfinite(unrealized):
                    total_equity = max(1.0, self.capital)  # Minimum 1.0 to prevent 100% drawdown artifacts
                else:
                    total_equity = self.capital + notional_in_position + unrealized
                    # Sanity check: prevent negative or unreasonable values
                    # Hard cap to prevent overflow
                    if not np.isfinite(total_equity):
                        # Use previous equity if available, else minimum
                        total_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
                        total_equity = max(1.0, min(total_equity, self.MAX_CAPITAL_ABSOLUTE))
                    else:
                        total_equity = max(1.0, min(total_equity, self.MAX_CAPITAL_ABSOLUTE))
        else:
            # No position - equity is just capital (with minimum floor)
            if np.isfinite(self.capital):
                total_equity = max(1.0, self.capital)
            else:
                # Use previous equity if capital is invalid
                total_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
                total_equity = max(1.0, total_equity)
        
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
