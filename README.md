# Mamba 4chan 2

## About

The Kek of Destiny, the next generation Mamba 4chan, is here.

## Installation

We provided a simple [setup.sh](setup.sh) to install the Conda environment. You need to satisfy the following prerequisites:

- Linuxf
- NVIDIA GPU
- CUDA 12+ supported GPU driver
- Miniforge

Then, simply run `source ./setup.sh` to get started.

## Dataset

We utilized the same preprocessed [Raiders of the Lost Kek dataset](https://arxiv.org/abs/2001.07487) detailed in the [original mamba 4chan repo](https://github.com/catalpaaa/Mamba-4chan). You can also find the download link there.

## Fine-tuned Models

We provide the following fine-tuned models, each trained for one epoch on the tokenized dataset using a single RTX 4090 with a context size of 2048 tokens and a batch size of 409,600 tokens. Mixed precision (bf16) was used for training, while the model weights were stored in fp32. We will release more models and improved versions as opportunities arise.

| Name               | Model Dim. | Num. of Layers | Attention Layers | Download                  | Fine-tuning Log |
|--------------------|------------|----------------|------------------|---------------------------|-----------------|
| Mamba 4chan 2 780M | 1536       | 48             | None             | [Download][780M download] | [log][780M log] |

[780M download]: https://archive.org/details/mamba_4chan_2_780m
[780M log]: https://wandb.ai/catalpa/Mamba%204chan%202%20780m

## Training and Inferencing

We provide [train.py](train.py), which contains all the necessary code to train a Mamba 4chan 2 model and log the training progress. The logged parameters can be modified in [model.py](model.py).

The base model's hyperparameters are stored in [model_config.py](model_config.py), and you can adjust them as needed. When further training our model, note that all hyperparameters are saved directly in the model file. For more information, refer to [PyTorch Lightning's documentation](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#contents-of-a-checkpoint). The same applies to inferencing, as PyTorch Lightning automatically handles all parameters when loading our model.

Here's a sample code snippet to perform inferencing with Mamba 4chan 2:

```python
from transformers import AutoTokenizer

from model import mamba_4chan

model = mamba_4chan.load_from_checkpoint("path_to.ckpt")

# from model_config import ssm_780m_config
# model = mamba_4chan.load_from_checkpoint(
#     "path_to_weight_only.ckpt",
#     config = ssm_780m_config()
# )

model.cuda()
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
text = "-----\n\n--- 943264000\nOur country".strip()
pred = model.generate_text(tokenizer, text, 512)
```

You can also use this [colab notebook](https://colab.research.google.com/drive/1buezmpgw30JahWplErA8GkkyhS7bgbFI?usp=sharing) for a quick demo.

## Credits

Our work builds upon the remarkable achievement of [Mamba](https://arxiv.org/abs/2312.00752) <3.
