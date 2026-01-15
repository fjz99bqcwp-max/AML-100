#!/usr/bin/env python3
"""Test script to debug gradient computation error."""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import asyncio

# Set working directory
os.chdir('/Users/nheosdisplay/VSC/AML/AML-100')

# Import the actual model directly
sys.path.insert(0, '/Users/nheosdisplay/VSC/AML/AML-100')
from src.ml_model import MLModel
from src.data_fetcher import DataFetcher

async def test_training():
    """Test the exact training path used in autonomous mode."""
    print("Creating ML model...")
    ml = MLModel('config/params.json')
    
    print(f"Model device: {ml.device}")
    
    # Fetch real data like the system does
    print("\nFetching training data...")
    fetcher = DataFetcher({})
    df = await fetcher.get_training_data(
        symbol="SPX",
        interval="1m",
        days=30,
        append_live=False
    )
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    
    try:
        print(f"\nStarting training (5 epochs)...")
        results = await ml.train(df, epochs=5)
        print(f"\nTraining completed! Results: {results}")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_training())
