<div align="center">

<h1> Retrieval Augmented Generation for Nursing Question Answering with Large Language Models </h1>


![colored_mesh (1)](img/logo03.png)
</div>

---


# Local LLM Deployment Project

This project provides a streamlined solution for locally deploying large language models such as ChatGLM2-6B and LLaMA-7B, integrated with a Chinese embedding model, suitable for local inference, knowledge-based QA, and other applications.

---

## 1. Prerequisites

### 1.1 System Requirements

#### Supported Models
- **ChatGLM2-6B**
- **LLaMA-2-7B**

#### Minimum GPU Memory
- **7GB VRAM**

#### Recommended GPUs
- NVIDIA RTX 3090
- NVIDIA RTX 3060 or higher

#### Software Environment
- **Python Version**: `>=3.10, <3.11` (Python 3.10 strongly recommended)
- **CUDA Version**: `>=11.7`

> ⚠️ **Note**: Ensure that CUDA and cuDNN are properly installed and configured with PyTorch GPU support.

---

### 1.2 Download Base LLMs

Please download the following models from Hugging Face to your local machine:

- **ChatGLM2-6B**:  
  [https://huggingface.co/zai-org/chatglm2-6b](https://huggingface.co/zai-org/chatglm2-6b)

- **LLaMA-2-7B**:  
  [https://huggingface.co/meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)

> 💡 **Tip**: You need to log in to Hugging Face and accept the model license agreement to download LLaMA series models.

---

### 1.3 Download Embedding Model

For Chinese text vectorization:

- **BAAI/bge-large-zh**  
  [https://huggingface.co/BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh)

It is recommended to store all models under the `models/` directory for easier management.

---

## 2. Deployment

### 2.1 Install Python Dependencies

```bash
pip install -r requirements.txt
If requirements.txt is not available, install the required packages manually:

bash
pip install transformers==4.33.3
pip install torch>=2.0.1
pip install torchvision
pip install torchaudio
pip install fastapi>=0.103.1
pip install nltk~=3.8.1

```

### 2.2 Launch the Project
After installation, start the service with:

```bash
python startup.py -a
```


#### 🐳Building With Docker Images
```shell
Coming soon
```


## Citation

If you use the codes and datasets , please cite the following paper(not published yet).

```
coming soon
```

