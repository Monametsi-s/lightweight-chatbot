"""A Modern LLM Chatbot with Chat Templates.

This module loads a lightweight causal language model (SmolLM2-360M-Instruct)
and runs an interactive conversational loop in the terminal using HuggingFace
chat templates and PyTorch.
"""

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    """Initialize the model and run the interactive chat session."""
    # Suppress non-critical warning messages
    warnings.filterwarnings("ignore")

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print("Loading model...")

    # Load tokenizer and configure padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.unk_token

    # Determine execution device (Apple Silicon GPU via MPS or fallback to CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load the causal language model onto the selected device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    # Initialize conversation history with a system instruction
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Give short and concise answers in 2-3 lines."
            ),
        }
    ]

    print("Chatbot started. Type 'exit' to quit.\n")

    # Start the interactive conversation loop
    while True:
        try:
            user_input = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chatbot.")
            break

        # Check for exit command
        if user_input.strip().lower() == "exit":
            break

        # Skip empty inputs
        if not user_input.strip():
            continue

        # Append user message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Retain system prompt and limit context to the last 10 messages
        messages = [messages[0]] + messages[-10:]

        # Format and tokenize conversation history using the chat template
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            max_length=512,
        )

        # Move token tensors to model's execution device
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

        # Generate response tokens without tracking gradients
        with torch.inference_mode():
            outputs = model.generate(
                tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=60,
                temperature=0.5,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode generated tokens to readable text (excluding input prompt)
        generated_tokens = outputs[0][tokenized["input_ids"].shape[-1]:]
        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        # Display the response
        print(f"Bot: {response}\n")
        messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()
