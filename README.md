# An LLM for Telemedicine KIOSK

<div align="center">
  <img src="https://github.com/srinivassake/telemedicine-kiosk/blob/main/Screenshot.png" width="200"/>
  
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Model Size](https://img.shields.io/badge/Model%20Size-3.87B%20params-brightgreen)](https://huggingface.co/savinirsekas/MediTalk-300-Mistral-7B-4bit)
  [![Quantization](https://img.shields.io/badge/Quantization-4bit-orange)](https://huggingface.co/savinirsekas/MediTalk-300-Mistral-7B-4bit)
</div>

## Overview

MediTalk-300 is a specialized Large Language Model (LLM) fine-tuned for telemedicine applications. Built on the Mistral-7B architecture and quantized to 4-bit precision, this model is optimized for medical conversations and patient interactions in a kiosk environment.

## Features

- 🏥 Specialized in medical conversations
- 💡 Fine-tuned on 1700+ synthetic conversational data
- 📚 Covers 300+ common diseases
- ⚡ 4-bit quantization for efficient deployment
- 🔧 Multiple deployment options

## Quick Start

### Using Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the model:
```bash
ollama pull savinirsekas/MediTalk-300-Mistral-7B-4bit
```
3. Run the model:
```bash
ollama run savinirsekas/MediTalk-300-Mistral-7B-4bit
```

### Using Open WebUI

1. Install Docker if you haven't already
2. Run the following command:
```bash
docker run -d --name open-webui -p 3000:8080 -v open-webui:/app/backend/data --restart always ghcr.io/open-webui/open-webui:main
```
3. Access the WebUI at `http://localhost:3000`
4. Add your model using the Hugging Face model ID: `savinirsekas/MediTalk-300-Mistral-7B-4bit`

### Using Docker

1. Pull the Docker image:
```bash
docker pull savinirsekas/MediTalk-300-Mistral-7B-4bit
```
2. Run the container:
```bash
docker run -d -p 8000:8000 savinirsekas/MediTalk-300-Mistral-7B-4bit
```

## Model Details

- **Base Model**: Mistral-7B-v0.3
- **Fine-tuned Version**: Mistral-7B-Instruct-v0.3
- **Quantization**: 4-bit (using bitsandbytes)
- **License**: Apache 2.0
- **Model Size**: 3.87B parameters

## Usage Examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "savinirsekas/MediTalk-300-Mistral-7B-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Example conversation
prompt = "What are the common symptoms of flu?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## System Requirements

- Minimum 8GB RAM
- CUDA-compatible GPU recommended
- 10GB free disk space

## License

This model is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this model in your research or project, please cite:

```bibtex
@misc{MediTalk-300-Mistral-7B-4bit,
  author = {savinirsekas},
  title = {MediTalk-300: A Fine-tuned Mistral-7B Model for Telemedicine},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/savinirsekas/MediTalk-300-Mistral-7B-4bit}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.