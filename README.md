# Lightweight Chatbot

A simple, lightweight conversational AI chatbot built with Python and Hugging Face's [Transformers](https://github.com/huggingface/transformers) library. It utilizes Meta's distilled conversational model ([`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill)) to deliver fast, natural multi-turn dialogue directly in the terminal without requiring heavy computational resources.

---

## Features

- ⚡ **Lightweight & Efficient**: Powered by `facebook/blenderbot-400M-distill`, optimized for low-latency local execution.
- 🧠 **Multi-Turn Context**: Maintains conversational history across turns for coherent, contextual interactions.
- 🎛️ **Tuned Generation**: Configured with temperature sampling, top-p truncation, repetition penalties, and n-gram blocking to prevent repetitive loops and enhance conversational quality.
- 💻 **Clean CLI Interface**: Interactive command-line chat experience.

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

Run the chatbot script from your terminal:

```bash
python chatbot.py
```

> **Note**: On the first run, Hugging Face will automatically download and cache the `facebook/blenderbot-400M-distill` model weights (~1.5 GB). Subsequent runs will load directly from the local cache.

---

## Project Structure

```
lightweight-chatbot/
├── chatbot.py          # Main chatbot inference script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Configuration & Customization

You can adjust generation parameters directly in `chatbot.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_new_tokens` | `60` | Maximum length of generated response tokens |
| `temperature` | `0.6` | Controls randomness (lower = more deterministic, higher = more creative) |
| `top_p` | `0.85` | Nucleus sampling threshold |
| `repetition_penalty` | `1.3` | Penalizes repeating words or phrases |
| `no_repeat_ngram_size` | `3` | Prevents the model from repeating 3-word phrases |

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
