"""
Prompt Template Layer
Responsibility: Converting structured chat messages into formatted prompt strings
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class ChatMessage:
    """Represents a single message in a conversation"""
    def __init__(self, role: str, content: str, content_type: str = "text"):
        self.role = role
        self.content = content
        self.content_type = content_type  # "text" or "image"
    
    def __repr__(self):
        return f"ChatMessage(role={self.role}, content_type={self.content_type})"


class PromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    @abstractmethod
    def render(self, messages: List[ChatMessage]) -> str:
        """Convert list of messages into a formatted prompt string"""
        pass


class RadiologyTemplate(PromptTemplate):
    """Template specifically designed for radiology CT report generation"""
    
    def __init__(self, image_token: str = "<image>"):
        self.image_token = image_token
        self.system_prompt = (
            "You are an expert radiologist. Given CT scan findings, "
            "provide a clear and concise impression that summarizes the "
            "key diagnostic observations and clinical significance."
        )
    
    def render(self, messages: List[ChatMessage]) -> str:
        """
        Renders messages into the format:
        SYSTEM: {system_prompt}
        USER: {findings_text}
        USER: {image_token}
        ASSISTANT: 
        """
        prompt_parts = []
        
        # Always start with system message
        prompt_parts.append(f"SYSTEM: {self.system_prompt}")
        
        # Process user messages
        for message in messages:
            if message.role == "user":
                if message.content_type == "text":
                    prompt_parts.append(f"USER: {message.content}")
                elif message.content_type == "image":
                    prompt_parts.append(f"USER: {self.image_token}")
            elif message.role == "assistant":
                prompt_parts.append(f"ASSISTANT: {message.content}")
        
        # Add assistant prompt if not already present
        if not any(msg.role == "assistant" for msg in messages):
            prompt_parts.append("ASSISTANT:")
        
        return "\n".join(prompt_parts)


class LLaVAStyleTemplate(PromptTemplate):
    """Template that mimics LLaVA's conversation format"""
    
    def __init__(self, image_token: str = "<image>", 
                 system_prompt: Optional[str] = None):
        self.image_token = image_token
        self.system_prompt = system_prompt or (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions."
        )
    
    def render(self, messages: List[ChatMessage]) -> str:
        """
        Renders in LLaVA v1 style:
        {system_prompt}
        USER: {content} ASSISTANT: {response}
        """
        prompt_parts = [self.system_prompt]
        
        current_exchange = []
        for message in messages:
            if message.role == "user":
                if message.content_type == "image":
                    current_exchange.append(self.image_token)
                else:
                    current_exchange.append(message.content)
            elif message.role == "assistant":
                user_content = " ".join(current_exchange) if current_exchange else ""
                prompt_parts.append(f"USER: {user_content} ASSISTANT: {message.content}")
                current_exchange = []
        
        # Handle case where there's no assistant response yet
        if current_exchange:
            user_content = " ".join(current_exchange)
            prompt_parts.append(f"USER: {user_content} ASSISTANT:")
        
        return " ".join(prompt_parts)


class SimpleTemplate(PromptTemplate):
    """Simple, customizable template"""
    
    def __init__(self, 
                 system_prompt: str,
                 user_prefix: str = "USER:",
                 assistant_prefix: str = "ASSISTANT:",
                 image_token: str = "<image>",
                 separator: str = "\n"):
        self.system_prompt = system_prompt
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.image_token = image_token
        self.separator = separator
    
    def render(self, messages: List[ChatMessage]) -> str:
        prompt_parts = [f"SYSTEM: {self.system_prompt}"]
        
        for message in messages:
            if message.role == "user":
                content = self.image_token if message.content_type == "image" else message.content
                prompt_parts.append(f"{self.user_prefix} {content}")
            elif message.role == "assistant":
                prompt_parts.append(f"{self.assistant_prefix} {message.content}")
        
        return self.separator.join(prompt_parts)


# Factory function for easy template selection
def create_template(template_type: str = "radiology", **kwargs) -> PromptTemplate:
    """Factory function to create different template types"""
    if template_type == "radiology":
        return RadiologyTemplate(**kwargs)
    elif template_type == "llava":
        return LLaVAStyleTemplate(**kwargs)
    elif template_type == "simple":
        return SimpleTemplate(**kwargs)
    else:
        raise ValueError(f"Unknown template type: {template_type}")


if __name__ == "__main__":
    # Example usage
    template = RadiologyTemplate()
    
    messages = [
        ChatMessage("user", "Findings: Bilateral lower lobe consolidation with air bronchograms."),
        ChatMessage("user", "", "image"),
    ]
    
    prompt = template.render(messages)
    print("Generated prompt:")
    print(prompt)
    print()
    
    # Example with response
    messages_with_response = messages + [
        ChatMessage("assistant", "Impression: Bilateral pneumonia with characteristic air bronchograms.")
    ]
    
    prompt_with_response = template.render(messages_with_response)
    print("Prompt with response:")
    print(prompt_with_response)
