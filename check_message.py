import argparse
import os
from datasets import load_dataset
from PIL import Image
from config.configuration_latent import LatentConfig
# Import the formatting logic directly to test it without loading the full model/processor
from data.dataset_latent import LatentReasoningDataset

# A dummy wrapper to access the static methods or we can just instantiate the class partially
# or copy the formatting logic.
# To be most accurate, we should use the method from the class.

class DummyDataset:
    """Helper to expose formatting logic from LatentReasoningDataset without initialization overhead"""
    def _detect_type(self, name):
        if "M3CoT" in name: return "m3cot"
        if "LLaVA" in name: return "llava"
        return "scienceqa"
        
    def _format_text(self, item_wrapper):
        # Delegate to the actual class method logic by instantiating temporarily or copying.
        # Since I cannot import the class method if it uses 'self' for config, 
        # I will use a dummy instance.
        pass

def check_dataset_format(dataset_name, split, index=0):
    print(f"\n{'='*20} Checking {dataset_name} {'='*20}")
    try:
        ds = load_dataset(dataset_name, split=split)
        raw_item = ds[index]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use a dummy instance of LatentReasoningDataset to access _format_text
    # We need a dummy config for initialization, but _format_text doesn't use self.config
    # except for maybe some very specific flags? 
    # Actually _format_text in the provided code DOES NOT use self.config, only self.
    # But it does not use any self properties. It is purely functional based on item_wrapper.
    
    # Let's instantiate the real dataset class with None processor/config just to access the method
    # or better, copy the logic or subclass.
    # To avoid errors, I'll just use the class method by passing 'self' as a dummy object.
    
    dummy_self = type('Dummy', (object,), {})()
    
    # We need to re-implement _detect_type or bind it
    def _detect_type(name):
        if "M3CoT" in name: return "m3cot"
        if "LLaVA" in name: return "llava"
        return "scienceqa"
    
    ds_type = _detect_type(dataset_name)
    
    item_wrapper = {
        'raw_item': raw_item,
        'ds_type': ds_type
    }
    
    # Call the logic directly (simulating LatentReasoningDataset._format_text)
    # Since I cannot import the method if it's an instance method easily without instance,
    # I will invoke it via the class definition in dataset_latent.py if I could import it.
    # Since I pasted the code in dataset_latent.py, I can assume LatentReasoningDataset is importable.
    
    # Create a minimal instance
    class ConfigMock:
        pass
    
    # We instantiate the real class but bypass __init__ to avoid overhead/errors
    real_ds = LatentReasoningDataset.__new__(LatentReasoningDataset)
    
    prompt, answer = real_ds._format_text(item_wrapper)
    
    print(f"--- [TYPE]: {ds_type} ---")
    print(f"--- [PROMPT] ---\n{prompt}")
    print(f"\n--- [ANSWER] ---\n{answer}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    # You can specify which datasets to check here or just run default list
    args = parser.parse_args()

    # List of datasets to check based on your config
    datasets_to_check = [
        ("derek-thomas/ScienceQA", "train"),
        ("LightChen2333/M3CoT", "train"),
        # ("Xkev/LLaVA-CoT-100k", "train") # Optional, logic hasn't changed much
    ]

    for name, split in datasets_to_check:
        check_dataset_format(name, split, index=0)

if __name__ == "__main__":
    main()