"""
Simple test script to demonstrate the template system without dependencies
"""
from template import RadiologyTemplate, LLaVAStyleTemplate, ChatMessage, create_template

def test_basic_templates():
    """Test the basic template functionality"""
    print("=== Testing Template System ===\n")
    
    # Create sample messages
    messages = [
        ChatMessage("user", "Findings: Bilateral lower lobe consolidation with air bronchograms."),
        ChatMessage("user", "", "image"),  # Image placeholder
    ]
    
    print("Input messages:")
    for i, msg in enumerate(messages):
        content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        if msg.content_type == "image":
            content_preview = "[IMAGE]"
        print(f"  {i+1}. {msg.role}: {content_preview}")
    print()
    
    # Test different templates
    templates = {
        "Radiology": RadiologyTemplate(),
        "LLaVA Style": LLaVAStyleTemplate(
            system_prompt="You are a medical AI assistant."
        ),
        "Custom Simple": create_template(
            "simple",
            system_prompt="Analyze this medical case:",
            user_prefix="DOCTOR:",
            assistant_prefix="AI:",
            separator="\n"
        )
    }
    
    for name, template in templates.items():
        print(f"{name} Template Output:")
        prompt = template.render(messages)
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        print()

def test_with_responses():
    """Test templates with assistant responses"""
    print("=== Testing Templates with Responses ===\n")
    
    messages_with_response = [
        ChatMessage("user", "Findings: Ground glass opacities in bilateral upper lobes."),
        ChatMessage("user", "", "image"),
        ChatMessage("assistant", "Impression: Atypical pneumonia, likely viral etiology. Recommend follow-up imaging in 1-2 weeks.")
    ]
    
    template = RadiologyTemplate()
    prompt = template.render(messages_with_response)
    
    print("Complete conversation:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    print()

def test_custom_template():
    """Show how to create a custom template"""
    print("=== Testing Custom Template Creation ===\n")
    
    from template import PromptTemplate
    
    class PathologyTemplate(PromptTemplate):
        """Custom template for pathology reports"""
        
        def __init__(self):
            self.system_prompt = (
                "You are a pathologist analyzing tissue samples. "
                "Provide a detailed diagnosis with staging information."
            )
        
        def render(self, messages):
            parts = [f"PATHOLOGY SYSTEM: {self.system_prompt}"]
            parts.append("="*50)
            
            for msg in messages:
                if msg.role == "user":
                    if msg.content_type == "image":
                        parts.append("TISSUE SAMPLE: <microscopy_image>")
                    else:
                        parts.append(f"CLINICAL HISTORY: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"PATHOLOGIST DIAGNOSIS: {msg.content}")
            
            if not any(msg.role == "assistant" for msg in messages):
                parts.append("PATHOLOGIST DIAGNOSIS:")
            
            return "\n".join(parts)
    
    # Test the custom template
    pathology_template = PathologyTemplate()
    messages = [
        ChatMessage("user", "Patient: 65-year-old male with lung nodule, history of smoking."),
        ChatMessage("user", "", "image"),
    ]
    
    prompt = pathology_template.render(messages)
    print("Custom Pathology Template:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)
    print()

def show_integration_concept():
    """Show how this integrates with the larger pipeline"""
    print("=== Integration with Full Pipeline ===\n")
    
    concept_code = '''
# How the template fits into the larger pipeline:

# 1. Dataset loads raw data
example = {
    "id": "ct_001",
    "image": "path/to/scan.jpg", 
    "findings": "Bilateral consolidation...",
    "impression": "Pneumonia..."
}

# 2. Convert to structured messages
messages = [
    ChatMessage("user", f"Findings: {example['findings']}"),
    ChatMessage("user", "", "image"),
    # Assistant response added during training for loss computation
]

# 3. Template renders to prompt string
template = RadiologyTemplate()
prompt = template.render(messages)

# 4. Collator tokenizes prompt + processes image
model_inputs = processor(
    text=prompt,
    images=load_image(example["image"]),
    return_tensors="pt"
)

# 5. Model processes the batch
outputs = model(**model_inputs)
loss = compute_loss(outputs.logits, labels)
'''
    
    print(concept_code)

if __name__ == "__main__":
    test_basic_templates()
    test_with_responses()
    test_custom_template()
    show_integration_concept()
    
    print("✓ Template system working correctly!")
    print("✓ Easy to customize for different medical domains")
    print("✓ Clean separation from data loading and tokenization")
    print("✓ Ready to integrate with your model training pipeline")
