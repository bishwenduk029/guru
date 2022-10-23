# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPT-NeoX model

from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/gpt-neox-20b"
    ).half().cuda()
    print("done")

    print("downloading tokenizer...")
    GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    print("done")


if __name__ == "__main__":
    download_model()
