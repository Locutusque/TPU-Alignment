# TPU-Alignment
Fully fine-tune large models like Mistral-7B, Llama-2-13B, or Qwen-14B completely for free.

The code in this repository have powered some larger models on my [Hugging Face profile](https://huggingface.co/Locutusque). I recommend using TPUs provided for free by Kaggle, they are strong enough to train models of up to 7 billion parameters without freezing parameters. [SPMD](https://pytorch.org/xla/release/2.1/index.html#pytorch-xla-spmd-user-guide) is used as a parallelization technique for high MXU efficiency while training.

Not every model architecture is supported by this code, here's a complete list of models supported:
- llama
- mistral
- gpt2? (untested but should work)
- gptneox
- qwen2
- t5

I'm open to contributions that propose additional model architectures.

## Getting started
1. Head on over to Kaggle, make sure you have verified your account with a phone number, and create a new notebook. Select TPU VM v3-8 as the accelerator.
2. Upload the ```spmd_util.py``` file to the input data
3. Import the notebook ```fully-finetune-large-models-for-free.ipynb``` into the kaggle notebook.
4. Modify the notebook to your needs. Make sure to provide your hugging face write access token in the Kaggle secrets.
5. Click save version, then select "save and run all". Make sure the training run does not run longer than 9 hours or the process will be killed!
