# Medical Image Report Generation Pipeline

Clean, modular pipeline for training multimodal models on medical imaging tasks. Extracts the essential components from LLaVA without the complexity, with built-in MIMIC-CXR support.

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers Pillow pandas
```

### 2. Test with Sample Data
```bash
python demo.py
```

### 3. Use Real Medical Data (MIMIC-CXR)
```bash
# Download small test subset
python download_mimic_subset.py --username YOUR_PHYSIONET_USERNAME --num-patients 10

# Process data
python setup_mimic_cxr.py --data-dir mimic-cxr-subset

# Test with real radiology reports
python demo.py --use-mimic --mimic-data-dir mimic-cxr-subset
```

## Architecture

Three loosely-coupled components:

1. **Template Layer** (`template.py`) - Formats chat messages into prompts
2. **Dataset Layer** (`dataset.py`) - Loads images and medical reports  
3. **Collator Layer** (`collator.py`) - Batches and tokenizes for training

## Usage Example

```python
from template import RadiologyTemplate
from dataset import CTReportDataset
from collator import MultimodalCollator

# Load data
dataset = CTReportDataset("data.json", "images/", is_mimic=True)

# Set up template
template = RadiologyTemplate(image_token="<image>")

# Create training batch
collator = MultimodalCollator(processor, template)
dataloader = DataLoader(dataset, collate_fn=collator)

# Train
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs["loss"]
```

## MIMIC-CXR Dataset

Large dataset of chest X-rays with radiology reports (377K images, 228K reports).

**Requirements**: PhysioNet credentialed account + signed data use agreement

**Get Access**:
1. Register at https://physionet.org/register/
2. Complete credentialing process  
3. Apply for MIMIC-CXR access

**Download**:
- **Test Subset**: `python download_mimic_subset.py --username USERNAME`
- **Full Dataset**: `python setup_mimic_cxr.py --instructions`

## Files

| File | Purpose |
|------|---------|
| `template.py` | Prompt formatting (RadiologyTemplate, LLaVATemplate, etc.) |
| `dataset.py` | Data loading (supports MIMIC-CXR and custom formats) |
| `collator.py` | Batching and tokenization |
| `setup_mimic_cxr.py` | Process MIMIC-CXR data |
| `download_mimic_subset.py` | Download MIMIC-CXR test subset |
| `demo.py` | Pipeline demonstration |
| `training.py` | Training integration example |

## Data Format

```json
[
  {
    "id": "example_001", 
    "image": "scan.jpg",
    "findings": "Bilateral lower lobe consolidation...",
    "impression": "Pneumonia with complications..."
  }
]
```

## Key Benefits

✅ **No LLaVA Dependencies** - Only essential packages  
✅ **Medical-Focused** - Radiology templates and data structures  
✅ **Real Data Ready** - MIMIC-CXR integration built-in  
✅ **Modular Design** - Swap any component independently  
✅ **Production Ready** - Memory efficient, proper error handling  

## Next Steps

1. **Test Pipeline**: Start with `python demo.py`
2. **Get MIMIC-CXR**: Follow setup instructions for real data
3. **Customize**: Modify templates for your medical domain
4. **Integrate**: Connect with your model (MedGemma, LLaVA, etc.)
5. **Scale**: Train on full MIMIC-CXR dataset

Perfect for training medical AI models without LLaVA's complexity!
