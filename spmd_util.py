import torch
import torch.nn as nn
import re
import torch_xla.experimental.xla_sharding as xs
import torch_xla.core.xla_model as xm
from transformers import (
    GPTNeoXConfig, T5Config, LlamaConfig, GPT2Config, MistralConfig, Qwen2Config, MixtralConfig, PhiConfig
)

# ends with $ to prevent sharding lora parameters


T5_RULES = (
    # embeddings
    ("shared$", ("mp", "fsdp")),
    ("embed_tokens$", ("mp", "fsdp")),
    
    # attention
    ("q$", ("fsdp", "mp")),
    ("k$", ("fsdp", "mp")),
    ("v$", ("fsdp", "mp")),
    ("o$", ("mp", "fsdp")),

    # mlp
    ("w$", ("fsdp", "mp")),
    ("wi_0$", ("fsdp", "mp")),
    ("wi_1$", ("fsdp", "mp")),
    ("wo$", ("mp", "fsdp")),

    # seq2seq lm head
    ("lm_head", ("fsdp", "mp")),
)

QWEN_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    )
GPT2_RULES = (
    # embeddings
    ("wte", ("mp", "fsdp")), 
    ("wpe", ("mp", "fsdp")),
    
    # attention
    ("c_attn", ("fsdp", "mp")),
    ("c_proj", ("mp", "fsdp")),
    
    # mlp
    ("c_fc", ("fsdp", "mp")), 
    ("c_proj", ("mp", "fsdp")),
    
    # output 
    ("ln_f", ("fsdp", "mp")),
)
MISTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    )


PHI_RULES = (
    ### (regex) linear modules, (list[sharding methods]) )
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.dense", ("mp", "fsdp")),
    ("mlp\\.fc2", ("mp", "fsdp")),  
    ("mlp\\.fc1", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    
)

LLAMA_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
    )

GPTNEOX_RULES = (
    # embeddings
    ("gpt_neox\\.embed_in", ("mp", "fsdp")),
    # atention
    ("attention\\.query_key_value$", ("fsdp", "mp")),
    ("attention\\.dense$", ("mp", "fsdp")),
    # mlp
    ("mlp\\.dense_h_to_4h$", ("fsdp", "mp")),
    ("mlp\\.dense_4h_to_h$", ("mp", "fsdp")),
    # output
    ("embed_out", ("fsdp", "mp")),
)



MIXTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("w1", ("fsdp", "mp")),
    ("w2", ("mp", "fsdp")),
    ("w3", ("fsdp", "mp")),
    ("gate", ("mp", "fsdp")),
    ("lm_head", ("fsdp", "mp")),
    )


    
ALL_RULES = [
    (GPTNeoXConfig, GPTNEOX_RULES),
    (T5Config, T5_RULES),
    (LlamaConfig, LLAMA_RULES),
    (GPT2Config, GPT2_RULES),
    (MistralConfig, MISTRAL_RULES),
    (Qwen2Config, QWEN_RULES),
    (MixtralConfig, MIXTRAL_RULES),
    (PhiConfig,PHI_RULES),
]

def find_rule(model):
    for config, rule in ALL_RULES:
        if model.config.__class__ == config:
            return rule
    raise Exception("unsupported model to partitioning")

strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "mp": 2
}

def partition_module(model, mesh, device=xm.xla_device(), verbose=False):
    partition_specs = find_rule(model)
    rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in partition_specs]
        
    # print(rule)

    for name, module in model.named_modules():
        module.to(device)
        # print(name, module.__class__.__name__)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in rule:
                if re.findall(rule_pattern, name):
                    if verbose:
                        print("match", rule_pattern, name)
                    
                    xs.mark_sharding(module.weight, mesh, spec)
                    break
        
def partition_module_dp(model, mesh, device=xm.xla_device(), verbose=False):
    spec = (1, 2)

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            xs.mark_sharding(module.weight, mesh, spec)
