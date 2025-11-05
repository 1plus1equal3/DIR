"""
Test script for VMambaBlock integration with config-driven model builder.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rectify.model import CustomModel
from rectify.config.vmamba_test import model as vmamba_config

def test_vmamba_model():
    """Test building and running VMamba model from config."""
    print("=" * 60)
    print("Testing VMamba Model Build")
    print("=" * 60)

    # Build model from config
    print("\n1. Building model from config...")
    try:
        model = CustomModel(vmamba_config)
        print("✓ Model built successfully!")
        print(f"\nModel structure:")
        print(model)
    except Exception as e:
        print(f"✗ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        input_tensor = torch.randn(batch_size, channels, height, width)
        print(f"Input shape: {input_tensor.shape}")

        # Forward pass
        with torch.no_grad():
            output_tensor = model(input_tensor)
        print(f"Output shape: {output_tensor.shape}")
        print("✓ Forward pass successful!")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_vmamba_model()
    sys.exit(0 if success else 1)
