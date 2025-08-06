"""
Collation Layer
Responsibility: Taking batches of raw examples, rendering them through templates,
and tokenizing everything into model-ready tensors.
"""
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Union, Callable
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

from template import PromptTemplate, ChatMessage
from dataset import CTReportExample


class MultimodalCollator:
    """
    Collator that handles text templating, tokenization, and image processing
    for multimodal models like LLaVA or MedGemma.
    """
    
    def __init__(self,
                 processor: Any,  # HuggingFace processor (handles both text and images)
                 template: PromptTemplate,
                 max_length: int = 2048,
                 image_token: str = "<image>",
                 ignore_index: int = -100):
        """
        Args:
            processor: HuggingFace processor that handles tokenization and image processing
            template: Template for rendering chat messages into prompts
            max_length: Maximum sequence length
            image_token: Token used as placeholder for images in text
            ignore_index: Index to ignore in loss computation (for padding tokens)
        """
        self.processor = processor
        self.template = template
        self.max_length = max_length
        self.image_token = image_token
        self.ignore_index = ignore_index
        
        # Get tokenizer from processor if available
        self.tokenizer = getattr(processor, 'tokenizer', processor)
    
    def __call__(self, batch: List[CTReportExample]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples into model inputs.
        
        Args:
            batch: List of CTReportExample objects
            
        Returns:
            Dictionary with keys: pixel_values, input_ids, attention_mask, labels
        """
        # Separate prompts and labels
        prompt_messages = []
        label_texts = []
        images = []
        
        for example in batch:
            # Get messages without the assistant response for prompt
            messages_for_prompt = example.to_messages(include_response=False)
            prompt_messages.append(messages_for_prompt)
            
            # Get the ground truth impression for labels
            label_texts.append(f"Impression: {example.impression}")
            
            # Collect images
            images.append(example.image)
        
        # Render prompts using template
        prompts = []
        for messages in prompt_messages:
            prompt_text = self.template.render(messages)
            prompts.append(prompt_text)
        
        # Process images and text together
        model_inputs = self._process_inputs(prompts, images)
        
        # Process labels separately
        labels = self._process_labels(label_texts)
        
        # Add labels to model inputs
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def _process_inputs(self, prompts: List[str], images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Process prompts and images into model inputs"""
        try:
            # Try to use processor for both text and images
            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
        except Exception:
            # Fallback: process separately
            # Process text
            text_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Process images
            if hasattr(self.processor, 'image_processor'):
                image_inputs = self.processor.image_processor(
                    images,
                    return_tensors="pt"
                )
                inputs = {**text_inputs, **image_inputs}
            else:
                # Simple fallback - just return text inputs
                inputs = text_inputs
                # Add dummy pixel values
                inputs["pixel_values"] = torch.zeros((len(images), 3, 224, 224))
        
        return inputs
    
    def _process_labels(self, label_texts: List[str]) -> torch.Tensor:
        """Process label texts into token IDs for loss computation"""
        # Tokenize labels
        label_inputs = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        labels = label_inputs["input_ids"]
        
        # Replace padding tokens with ignore_index
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        return labels


class SimpleCollator:
    """
    Simplified collator for cases where you don't have a full HuggingFace processor
    """
    
    def __init__(self,
                 tokenizer: Any,
                 template: PromptTemplate,
                 image_processor: Optional[Callable] = None,
                 max_length: int = 2048,
                 ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.template = template
        self.image_processor = image_processor
        self.max_length = max_length
        self.ignore_index = ignore_index
    
    def __call__(self, batch: List[CTReportExample]) -> Dict[str, torch.Tensor]:
        """Simple collation without HuggingFace processor"""
        # Render prompts
        prompts = []
        labels = []
        images = []
        
        for example in batch:
            # Get prompt messages (without response)
            prompt_messages = example.to_messages(include_response=False)
            prompt = self.template.render(prompt_messages)
            prompts.append(prompt)
            
            # Get label
            labels.append(f"Impression: {example.impression}")
            
            # Get image
            images.append(example.image)
        
        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Tokenize labels
        label_inputs = self.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Process labels for loss computation
        processed_labels = label_inputs["input_ids"].clone()
        processed_labels[processed_labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        # Process images if processor available
        if self.image_processor:
            pixel_values = self.image_processor(images)
        else:
            # Dummy pixel values
            pixel_values = torch.zeros((len(images), 3, 224, 224))
        
        return {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "pixel_values": pixel_values,
            "labels": processed_labels
        }


class DebugCollator:
    """Collator that prints debug information"""
    
    def __init__(self, base_collator):
        self.base_collator = base_collator
    
    def __call__(self, batch):
        print(f"\n=== Collating batch of {len(batch)} examples ===")
        
        for i, example in enumerate(batch):
            print(f"\nExample {i}:")
            print(f"  ID: {example.id}")
            print(f"  Findings: {example.findings[:100]}...")
            print(f"  Impression: {example.impression[:100]}...")
            print(f"  Image size: {example.image.size}")
        
        # Call the base collator
        result = self.base_collator(batch)
        
        print(f"\nCollated result shapes:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("=== End batch ===\n")
        
        return result


def create_dataloader(dataset,
                      collator,
                      batch_size: int = 4,
                      shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
    """Create a DataLoader with the specified collator"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )


if __name__ == "__main__":
    # Demo usage (would need actual tokenizer/processor)
    from dataset import CTReportDataset, create_sample_data
    from template import RadiologyTemplate
    
    # Create sample data
    json_path, images_dir = create_sample_data("sample_data", num_samples=5)
    
    # Create dataset
    dataset = CTReportDataset(json_path, images_dir)
    
    # Create template
    template = RadiologyTemplate()
    
    # For demo, we'll create a mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            
        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            max_len = 50
            input_ids = []
            attention_mask = []
            
            for text in texts:
                # Convert to token IDs (mock)
                tokens = [i % 1000 for i in range(len(text.split()))]
                
                # Pad or truncate
                if len(tokens) < max_len:
                    tokens.extend([self.pad_token_id] * (max_len - len(tokens)))
                    mask = [1] * len([t for t in tokens if t != self.pad_token_id]) + [0] * (max_len - len([t for t in tokens if t != self.pad_token_id]))
                else:
                    tokens = tokens[:max_len]
                    mask = [1] * max_len
                
                input_ids.append(tokens)
                attention_mask.append(mask)
            
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask)
            }
    
    # Create mock collator
    tokenizer = MockTokenizer()
    collator = SimpleCollator(tokenizer, template)
    debug_collator = DebugCollator(collator)
    
    # Create dataloader
    dataloader = create_dataloader(dataset, debug_collator, batch_size=2)
    
    # Test one batch
    print("Testing collator with sample batch...")
    batch = next(iter(dataloader))
    print(f"Final batch keys: {list(batch.keys())}")
