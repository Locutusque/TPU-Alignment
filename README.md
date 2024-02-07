# TPU-Alignment
Fully fine-tune large models like Mistral-7B, Llama-2-13B, or Qwen-14B completely for free.

The code in this repository have powered some larger models on my ![Hugging Face profile](https://huggingface.co/Locutusque). I recommend using TPUs provided for free by Kaggle, they are strong enough to train models of up to 7 billion parameters without freezing parameters. ![SPMD](https://pytorch.org/xla/release/2.1/index.html#pytorch-xla-spmd-user-guide) is used as a parallelization technique for high MXU efficiency while training.

Not every model architecture is supported by this code, here's a complete list of models supported:
- llama
- mistral
- gpt2? (untested but should work)
- gptneox
- qwen2
- t5
I'm open to contributions that propose additional model architectures.
