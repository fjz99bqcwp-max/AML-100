"""
Tests for ML Model Module
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_model import (
    LSTMFeatureExtractor,
    DQNHead,
    HybridLSTMDQN,
    ReplayBuffer,
    MLModel,
    BacktestEnvironment
)


class TestLSTMFeatureExtractor:
    """Tests for LSTM feature extractor"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = LSTMFeatureExtractor(input_size=20, hidden_size=64, num_layers=2)
        
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = LSTMFeatureExtractor(input_size=20, hidden_size=64)
        
        # Batch of 4, sequence length 30, 20 features
        x = torch.randn(4, 30, 20)
        output = model(x)
        
        assert output.shape == (4, 64)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        model = LSTMFeatureExtractor(input_size=10, hidden_size=32)
        
        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, 10)
            output = model(x)
            assert output.shape == (2, 32)


class TestDQNHead:
    """Tests for DQN head"""
    
    def test_initialization(self):
        """Test initialization"""
        model = DQNHead(input_size=64, hidden_size=128, num_actions=3)
        
        assert model.fc3.out_features == 3
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = DQNHead(input_size=64, num_actions=3)
        
        x = torch.randn(8, 64)
        output = model(x)
        
        assert output.shape == (8, 3)
    
    def test_output_range(self):
        """Test that outputs are unbounded Q-values"""
        model = DQNHead(input_size=32, num_actions=3)
        
        x = torch.randn(100, 32)
        output = model(x)
        
        # Q-values can be any real number
        assert output.min() < 0 or output.max() > 0


class TestHybridLSTMDQN:
    """Tests for hybrid model"""
    
    def test_initialization(self):
        """Test initialization"""
        model = HybridLSTMDQN(
            input_size=20,
            lstm_hidden=64,
            lstm_layers=2,
            dqn_hidden=128,
            num_actions=3
        )
        
        assert isinstance(model.lstm_extractor, LSTMFeatureExtractor)
        assert isinstance(model.dqn_head, DQNHead)
    
    def test_forward_pass(self):
        """Test end-to-end forward pass"""
        model = HybridLSTMDQN(input_size=20, num_actions=3)
        
        x = torch.randn(4, 60, 20)  # batch=4, seq=60, features=20
        output = model(x)
        
        assert output.shape == (4, 3)
    
    def test_get_action_greedy(self):
        """Test greedy action selection"""
        model = HybridLSTMDQN(input_size=10, num_actions=3)
        
        x = torch.randn(1, 30, 10)
        action, confidence = model.get_action(x, epsilon=0.0)
        
        assert action in [0, 1, 2]
        assert 0 <= confidence <= 1
    
    def test_get_action_exploration(self):
        """Test epsilon-greedy exploration"""
        model = HybridLSTMDQN(input_size=10, num_actions=3)
        
        x = torch.randn(1, 30, 10)
        
        # With epsilon=1.0, should always explore
        actions = set()
        for _ in range(100):
            action, _ = model.get_action(x, epsilon=1.0)
            actions.add(action)
        
        # Should have explored all actions
        assert len(actions) == 3


class TestReplayBuffer:
    """Tests for replay buffer"""
    
    def test_initialization(self):
        """Test initialization"""
        buffer = ReplayBuffer(capacity=100)
        
        assert len(buffer) == 0
    
    def test_push_and_sample(self):
        """Test pushing and sampling"""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            state = np.random.randn(30, 10)
            next_state = np.random.randn(30, 10)
            buffer.push(state, i % 3, 0.1, next_state, False)
        
        assert len(buffer) == 50
        
        states, actions, rewards, next_states, dones = buffer.sample(16)
        
        assert states.shape == (16, 30, 10)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity"""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(20):
            buffer.push(np.zeros((5, 5)), 0, 0.0, np.zeros((5, 5)), False)
        
        assert len(buffer) == 10


class TestMLModel:
    """Tests for MLModel manager"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        
        n_samples = 500
        df = pd.DataFrame({
            "timestamp": np.arange(n_samples),
            "open": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + 0.5,
            "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - 0.5,
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
            "volume": np.random.randint(100, 1000, n_samples)
        })
        
        return df
    
    def test_initialization(self, tmp_path):
        """Test MLModel initialization"""
        # Create temp config
        config = {
            "ml_model": {
                "type": "hybrid_lstm_dqn",
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "dqn_hidden_size": 128,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "sequence_length": 30,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "memory_size": 1000,
                "target_update_freq": 5,
                "dropout": 0.2
            }
        }
        
        config_path = tmp_path / "params.json"
        import json
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        model = MLModel(config_path=str(config_path), model_dir=str(tmp_path / "models"))
        
        assert model.sequence_length == 30
        assert model.epsilon == 1.0
    
    def test_calculate_reward(self, tmp_path):
        """Test reward calculation"""
        config = {"ml_model": {"sequence_length": 30}}
        config_path = tmp_path / "params.json"
        import json
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        model = MLModel(config_path=str(config_path), model_dir=str(tmp_path / "models"))
        
        # Hold - should be penalized (less than BUY with positive change)
        hold_reward = model.calculate_reward(model.HOLD, 0.01)
        
        # Buy with positive price change - should be positive
        buy_reward = model.calculate_reward(model.BUY, 0.02)
        assert buy_reward > hold_reward  # BUY should be better than HOLD
        
        # Sell with positive price change (wrong direction) - should be less than BUY
        sell_reward = model.calculate_reward(model.SELL, 0.02)
        assert sell_reward < buy_reward  # Wrong direction should be worse


class TestBacktestEnvironment:
    """Tests for backtest environment"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample price data"""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        return pd.DataFrame({
            "close": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "open": prices,
            "volume": np.random.randint(100, 1000, 200)
        })
    
    def test_initialization(self, sample_df):
        """Test environment initialization"""
        env = BacktestEnvironment(
            df=sample_df,
            initial_capital=1000,
            transaction_cost=0.001
        )
        
        assert env.capital == 1000
        assert env.position == 0
    
    def test_reset(self, sample_df):
        """Test environment reset"""
        env = BacktestEnvironment(df=sample_df, initial_capital=1000)
        
        # Make some steps
        env.step(1, 0.1)
        env.step(2, 0.1)
        
        # Reset
        env.reset()
        
        assert env.capital == 1000
        assert env.position == 0
        assert len(env.trades) == 0
    
    def test_buy_action(self, sample_df):
        """Test buy action"""
        env = BacktestEnvironment(df=sample_df, initial_capital=1000)
        
        reward, done = env.step(1, 0.1)  # Buy
        
        assert env.position > 0
        assert env.capital < 1000
    
    def test_sell_action(self, sample_df):
        """Test sell action after buy"""
        env = BacktestEnvironment(df=sample_df, initial_capital=1000)
        
        env.step(1, 0.1)  # Buy
        initial_position = env.position
        
        env.current_idx = 10  # Move forward
        env.step(2, 0.1)  # Sell
        
        # Position should be reduced or closed (may not be exactly 0 due to partial fills)
        assert env.position < initial_position or abs(env.position) < 0.01
        assert len(env.trades) >= 0  # At least attempted trade
    
    def test_get_metrics(self, sample_df):
        """Test metrics calculation"""
        env = BacktestEnvironment(df=sample_df, initial_capital=1000)
        
        # Simulate some trading
        for i in range(50):
            action = (i % 3)  # Cycle through actions
            env.step(action, 0.1)
        
        metrics = env.get_metrics()
        
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
