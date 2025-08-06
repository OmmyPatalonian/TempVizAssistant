"""
Integration Example: How to use the three components together
This demonstrates the complete pipeline from data loading to model training.
"""
import sys
import os
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent))

# Import components that don't need external dependencies
from template import RadiologyTemplate, LLaVAStyleTemplate, create_template

# Try to import components that need dependencies
try:
    from dataset import CTReportDataset, create_sample_data
    from collator import SimpleCollator, DebugCollator, create_dataloader
    HAS_FULL_DEPS = True
except ImportError:
    HAS_FULL_DEPS = False


def demo_pipeline(use_mimic=False, mimic_data_dir=None):
    """Demonstrate the complete three-component pipeline"""
    
    if not HAS_FULL_DEPS:
        print("Full pipeline demo requires torch and PIL dependencies")
        return None, None, None, None
    
    print("=== Medical Image Report Generation Pipeline Demo ===\n")
    
    if use_mimic and mimic_data_dir:
        # Use MIMIC-CXR data
        print("1. Loading MIMIC-CXR data...")
        json_path = os.path.join(mimic_data_dir, "processed", "train.json")
        images_dir = os.path.join(mimic_data_dir, "files")
        
        if not os.path.exists(json_path):
            print(f"   Error: MIMIC-CXR data not found at {json_path}")
            print("   Run: python setup_mimic_cxr.py --data-dir /path/to/mimic/data")
            return None, None, None, None
        
        dataset = CTReportDataset(json_path, images_dir, is_mimic=True)
        print(f"   Loaded {len(dataset)} MIMIC-CXR examples")
        
    else:
        # Create sample data
        print("1. Creating sample CT report data...")
        json_path, images_dir = create_sample_data("demo_data", num_samples=8)
        print(f"   Created data at: {json_path}")
        print(f"   Images at: {images_dir}")
        dataset = CTReportDataset(json_path, images_dir)
        print(f"   Loaded {len(dataset)} examples")
    
    # Show a sample
    example = dataset[0]
    print(f"   Sample finding: {example.findings[:80]}...")
    print(f"   Sample impression: {example.impression[:80]}...\n")
    
    # Step 3: Initialize the Template Layer
    print("2. Initializing Template Layer...")
    template = RadiologyTemplate(image_token="<image>")
    
    # Show template output
    messages = example.to_messages(include_response=False)
    prompt = template.render(messages)
    print("   Sample prompt generated:")
    print("   " + "\n   ".join(prompt.split("\n")))
    print()
    
    # Step 4: Initialize the Collation Layer
    print("3. Initializing Collation Layer...")
    
    # Mock tokenizer for demo (in real use, you'd use actual HuggingFace tokenizer)
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.vocab_size = 32000
            
        def __call__(self, texts, **kwargs):
            max_length = kwargs.get('max_length', 100)
            batch_size = len(texts)
            
            # Simple tokenization simulation
            try:
                import torch
                input_ids = torch.randint(1, self.vocab_size, (batch_size, max_length))
                attention_mask = torch.ones((batch_size, max_length))
            except ImportError:
                # Fallback without torch
                import random
                input_ids = [[random.randint(1, self.vocab_size-1) for _ in range(max_length)] 
                           for _ in range(batch_size)]
                attention_mask = [[1 for _ in range(max_length)] for _ in range(batch_size)]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
    
    tokenizer = MockTokenizer()
    collator = SimpleCollator(tokenizer, template, max_length=512)
    debug_collator = DebugCollator(collator)
    
    print("   Collator configured with mock tokenizer\n")
    
    # Step 5: Create DataLoader and process a batch
    print("4. Testing complete pipeline...")
    dataloader = create_dataloader(dataset, debug_collator, batch_size=3, shuffle=False)
    
    # Process one batch
    batch = next(iter(dataloader))
    
    print("5. Pipeline Results:")
    print(f"   Successfully processed batch with keys: {list(batch.keys())}")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: shape {value.shape}")
        else:
            print(f"   {key}: {type(value)}")
    
    return dataset, template, collator, batch


def demo_different_templates():
    """Show how different templates produce different outputs"""
    
    print("\n=== Template Comparison Demo ===\n")
    
    # Create a sample example
    from template import ChatMessage
    messages = [
        ChatMessage("user", "Findings: Bilateral lower lobe consolidation with air bronchograms."),
        ChatMessage("user", "", "image"),
    ]
    
    templates = {
        "Radiology": RadiologyTemplate(),
        "LLaVA Style": LLaVAStyleTemplate(
            system_prompt="You are a medical AI assistant specialized in radiology."
        ),
        "Simple": create_template("simple", 
                                 system_prompt="Analyze the CT scan and provide diagnosis.",
                                 separator="\n")
    }
    
    for name, template in templates.items():
        print(f"{name} Template Output:")
        prompt = template.render(messages)
        print("   " + "\n   ".join(prompt.split("\n")))
        print()


def demo_extensibility():
    """Show how the modular design enables easy extensions"""
    
    print("\n=== Extensibility Demo ===\n")
    
    # Custom template for a different medical domain
    from template import PromptTemplate, ChatMessage
    
    class CardiologyTemplate(PromptTemplate):
        def __init__(self):
            self.system_prompt = (
                "You are a cardiologist analyzing cardiac imaging. "
                "Provide differential diagnoses and recommend next steps."
            )
        
        def render(self, messages):
            parts = [f"CARDIOLOGY SYSTEM: {self.system_prompt}"]
            
            for msg in messages:
                if msg.role == "user":
                    if msg.content_type == "image":
                        parts.append("CARDIAC IMAGE: <image>")
                    else:
                        parts.append(f"CLINICAL DATA: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"CARDIOLOGIST: {msg.content}")
            
            if not any(msg.role == "assistant" for msg in messages):
                parts.append("CARDIOLOGIST:")
            
            return "\n".join(parts)
    
    # Test the custom template
    cardio_template = CardiologyTemplate()
    messages = [
        ChatMessage("user", "ECG shows ST elevation in leads II, III, aVF"),
        ChatMessage("user", "", "image"),
    ]
    
    prompt = cardio_template.render(messages)
    print("Custom Cardiology Template:")
    print("   " + "\n   ".join(prompt.split("\n")))
    print("\nThis shows how easy it is to create domain-specific templates!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo medical image report generation pipeline")
    parser.add_argument("--use-mimic", action="store_true", help="Use MIMIC-CXR dataset")
    parser.add_argument("--mimic-data-dir", type=str, help="Path to MIMIC-CXR data directory")
    
    args = parser.parse_args()
    
    # For the demo, we need to handle the missing dependencies gracefully
    if HAS_FULL_DEPS:
        try:
            # Run the main demo
            dataset, template, collator, batch = demo_pipeline(args.use_mimic, args.mimic_data_dir)
            
            if dataset is not None:
                # Show template variations
                demo_different_templates()
                
                # Show extensibility
                demo_extensibility()
                
                print("\n=== Demo Complete ===")
                print("The three-component architecture provides:")
                print("✓ Clean separation of concerns")
                print("✓ Easy template customization")
                print("✓ Flexible data loading")
                print("✓ Efficient batching and tokenization")
                print("✓ Simple extension for new domains")
            
        except Exception as e:
            print(f"Demo error: {e}")
            print("Running limited demo...")
            demo_different_templates()
            demo_extensibility()
        
    else:
        print("Demo limited due to missing dependencies (torch, PIL)")
        print("Install them to run full demo: pip install torch transformers Pillow")
        print()
        
        # Still show the template demo which doesn't need external deps
        demo_different_templates()
        demo_extensibility()
