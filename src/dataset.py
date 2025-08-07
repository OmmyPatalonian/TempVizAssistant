"""
Dataset stuff for loading medical images and reports
"""
import json
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from .template import ChatMessage


class CTReportExample:
    #single example with image and text
    
    def __init__(self, 
                 example_id: str,
                 image_path: str,
                 findings: str,
                 impression: str,
                 image: Optional[Image.Image] = None):
        self.id = example_id
        self.image_path = image_path
        self.findings = findings
        self.impression = impression
        self._image = image
    
    @property
    def image(self) -> Image.Image:
        #lazy load image
        if self._image is None:
            self._image = Image.open(self.image_path).convert('RGB')
        return self._image
    
    def to_messages(self, include_response: bool = True) -> List[ChatMessage]:
        #convert to chat messages
        messages = [
            ChatMessage("user", f"Findings: {self.findings}"),
            ChatMessage("user", "", "image"),  #image placeholder
        ]
        
        if include_response:
            messages.append(ChatMessage("assistant", f"Impression: {self.impression}"))
        
        return messages
    
    def __repr__(self):
        return f"CTReportExample(id={self.id}, findings_len={len(self.findings)})"


class CTReportDataset(Dataset):
    #dataset for medical reports 
    
    def __init__(self, 
                 json_path: str,
                 image_folder: str,
                 load_images: bool = False):
        self.json_path = Path(json_path)
        self.image_folder = Path(image_folder)
        self.load_images = load_images
        
        # load json data
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        #validate structure
        self._validate_data()
        
        # preload images if needed
        if self.load_images:
            self._preload_images()
    
    def _validate_data(self):
        #check json has required fields
        if not isinstance(self.data, list):
            raise ValueError("JSON should contain a list of examples")
        
        required_fields = {"id", "image", "findings", "impression"}
        for i, item in enumerate(self.data):
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(f"Item {i} missing required fields: {missing_fields}")
    
    def _preload_images(self):
        #load all images to memory
        print(f"Preloading {len(self.data)} images...")
        for item in self.data:
            image_path = self.image_folder / item["image"]
            item["_preloaded_image"] = Image.open(image_path).convert('RGB')
        print("Images preloaded successfully")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> CTReportExample:
        #get single example
        item = self.data[idx]
        
        # build image path
        image_path = self.image_folder / item["image"]
        
        #use preloaded image if available
        image = item.get("_preloaded_image", None)
        
        return CTReportExample(
            example_id=item["id"],
            image_path=str(image_path),
            findings=item["findings"],
            impression=item["impression"],
            image=image
        )
    
    def get_messages(self, idx: int, include_response: bool = True) -> List[ChatMessage]:
        #get messages for specific example
        example = self[idx]
        return example.to_messages(include_response=include_response)
    
    def get_batch_messages(self, indices: List[int], include_response: bool = True) -> List[List[ChatMessage]]:
        # get messages for batch of examples
        return [self.get_messages(idx, include_response) for idx in indices]


class MultiDatasetWrapper:
    """Wrapper to handle multiple datasets (e.g., CT + MRI)"""
    
    def __init__(self, datasets: Dict[str, CTReportDataset]):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        
        # Calculate cumulative lengths for indexing
        self.lengths = [len(ds) for ds in datasets.values()]
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)
    
    def __len__(self) -> int:
        return sum(self.lengths)
    
    def __getitem__(self, idx: int) -> tuple[str, CTReportExample]:
        """Returns (dataset_name, example) tuple"""
        for i, cumlen in enumerate(self.cumulative_lengths):
            if idx < cumlen:
                dataset_name = self.dataset_names[i]
                local_idx = idx - (self.cumulative_lengths[i-1] if i > 0 else 0)
                return dataset_name, self.datasets[dataset_name][local_idx]
        
        raise IndexError(f"Index {idx} out of range")


def create_sample_data(output_dir: str, num_samples: int = 10):
    """Create sample CT report data for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sample findings and impressions
    sample_data = []
    findings_templates = [
        "Bilateral lower lobe consolidation with air bronchograms",
        "Ground glass opacities in the upper lobes bilaterally", 
        "Nodular opacity in the right middle lobe measuring 2.3 cm",
        "Pleural effusion on the left side with associated atelectasis",
        "Tree-in-bud pattern in bilateral lower lobes",
        "Honeycombing pattern in bilateral lower lobes consistent with fibrosis",
        "Mediastinal lymphadenopathy with nodes measuring up to 1.5 cm",
        "Pulmonary embolus in the right main pulmonary artery",
        "Pneumothorax on the right side, approximately 20%",
        "Multiple pulmonary nodules scattered throughout both lungs"
    ]
    
    impression_templates = [
        "Bilateral pneumonia",
        "Atypical pneumonia, possibly viral etiology", 
        "Suspicious pulmonary nodule, recommend follow-up CT in 3 months",
        "Left pleural effusion with compressive atelectasis",
        "Bronchiolar infection, likely infectious bronchiolitis",
        "Idiopathic pulmonary fibrosis",
        "Mediastinal lymphadenopathy, differential includes sarcoidosis",
        "Acute pulmonary embolism",
        "Spontaneous pneumothorax",
        "Multiple pulmonary metastases"
    ]
    
    for i in range(num_samples):
        sample_data.append({
            "id": f"ct_report_{i:03d}",
            "image": f"ct_scan_{i:03d}.jpg",
            "findings": findings_templates[i % len(findings_templates)],
            "impression": impression_templates[i % len(impression_templates)]
        })
    
    # Save JSON
    json_path = output_dir / "ct_reports.json"
    with open(json_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # Create dummy images (just colored squares for testing)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        # Create a simple colored image
        img = Image.new('RGB', (256, 256), color=(i*25 % 255, (i*50) % 255, (i*75) % 255))
        img.save(images_dir / f"ct_scan_{i:03d}.jpg")
    
    print(f"Created sample data in {output_dir}")
    print(f"JSON file: {json_path}")
    print(f"Images folder: {images_dir}")
    
    return str(json_path), str(images_dir)


if __name__ == "__main__":
    # Create sample data
    json_path, images_dir = create_sample_data("sample_data")
    
    # Test the dataset
    dataset = CTReportDataset(json_path, images_dir)
    
    print(f"Dataset length: {len(dataset)}")
    print()
    
    # Test getting an example
    example = dataset[0]
    print(f"Example: {example}")
    print(f"Findings: {example.findings}")
    print(f"Impression: {example.impression}")
    print(f"Image size: {example.image.size}")
    print()
    
    # Test getting messages
    messages = dataset.get_messages(0)
    print("Messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content} ({msg.content_type})")
    print()
    
    # Test batch messages
    batch_messages = dataset.get_batch_messages([0, 1, 2])
    print(f"Batch of {len(batch_messages)} message lists created")
