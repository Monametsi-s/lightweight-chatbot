# Lightweight Chatbot

A simple, fast, and lightweight conversational AI repository built with Python, PyTorch, and Hugging Face's [Transformers](https://github.com/huggingface/transformers) library.

This repository provides two distinct chatbot implementations designed for low-latency local execution directly in the terminal without requiring heavy computational resources:
1. **Modern LLM with Chat Templates (`chatbot_llm.py`)**: Powered by [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), utilizing standard chat templates, system prompt instructions, and Apple Silicon MPS / CPU hardware acceleration.
2. **Seq2Seq Dialogue Bot (`chatbot.py`)**: Powered by Meta's distilled conversational model ([`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill)).

---

## Features

- ⚡ **Lightweight & Efficient**: Compact model architectures (~360M–400M parameters) optimized for rapid inference on consumer hardware.
- 🚀 **Hardware Acceleration**: Automatic device detection supporting Apple Silicon GPU (`mps`) with seamless `cpu` fallback.
- 💬 **Hugging Face Chat Templates**: Utilizes tokenizer `apply_chat_template` formatting for system, user, and assistant roles.
- 🧠 **Multi-Turn Context**: Maintains conversational history across dialogue turns for coherent, context-aware interactions.
- 🎛️ **Fine-Tuned Generation**: Tailored sampling settings (`temperature`, `top_p`, `repetition_penalty`, `no_repeat_ngram_size`) to eliminate repetitive loops and deliver natural responses.
- 🛡️ **Clean & Robust Codebase**: Fully formatted per PEP 8 standards and rated **10.00 / 10** in static code analysis (Pylint & Flake8).

---

## Prerequisites

- **Python 3.8+**
- Virtual environment tool (`venv`, `conda`, etc.)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Monametsi-s/lightweight-chatbot.git
   cd lightweight-chatbot
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Modern LLM Chatbot (SmolLM2-360M-Instruct)
Runs the causal language model with chat template support and system instructions:

```bash
python chatbot_llm.py
```

### 2. Seq2Seq Chatbot (BlenderBot-400M)
Runs the sequence-to-sequence conversational bot:

```bash
python chatbot.py
```

> **Note**: On the first run of either script, Hugging Face will automatically download and cache the model weights to `~/.cache/huggingface/`. Subsequent runs will load directly from the local cache.

---

## Project Structure

```text
lightweight-chatbot/
├── chatbot.py           # Seq2Seq conversational chatbot (BlenderBot-400M)
├── chatbot_llm.py       # Modern LLM chatbot with Chat Templates (SmolLM2-360M)
├── requirements.txt     # Project dependencies
├── additional_notes.txt # Parameter references and generation notes
└── README.md            # Project documentation
```

---

## Configuration & Generation Parameters

Both chatbot implementations allow fine-tuning generation hyperparameters:

| Parameter | Default (`chatbot_llm.py`) | Default (`chatbot.py`) | Description |
| :--- | :--- | :--- | :--- |
| `max_new_tokens` | `60` | `60` | Maximum number of tokens generated per response |
| `temperature` | `0.5` | `0.6` | Controls randomness (lower = more deterministic, higher = more creative) |
| `top_p` | `0.80` | `0.85` | Nucleus sampling probability threshold |
| `repetition_penalty` | `1.3` | `1.3` | Penalizes repeated words and phrases |
| `no_repeat_ngram_size` | `3` | `3` | Prevents the model from repeating 3-word n-grams |
| `do_sample` | `True` | `True` | Enables probabilistic token sampling |

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
