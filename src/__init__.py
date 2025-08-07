"""
Medical Image Report Generation Pipeline

A clean, modular pipeline for training multimodal models on medical imaging tasks.
Extracts the essential components from LLaVA without the complexity.
"""

from .template import RadiologyTemplate, LLaVAStyleTemplate, SimpleTemplate, ChatMessage, create_template
from .dataset import CTReportDataset, CTReportExample, create_sample_data
from .collator import MultimodalCollator, SimpleCollator, create_dataloader

__version__ = "1.0.0"
__all__ = [
    # Templates
    "RadiologyTemplate",
    "LLaVAStyleTemplate", 
    "SimpleTemplate",
    "ChatMessage",
    "create_template",
    
    # Dataset
    "CTReportDataset",
    "CTReportExample",
    "create_sample_data",
    
    # Collator
    "MultimodalCollator",
    "SimpleCollator",
    "create_dataloader",
]
