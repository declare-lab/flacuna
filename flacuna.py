import torch
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from peft_flacuna import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules = ["q_proj", "v_proj"]
    lora_weight_path: str = ""
    bias: str = "none"
    

class FLACUNA:
    def __init__(self, model_path, device_id=0):
        
        lora_args = LoraArguments
        base_model = "TheBloke/vicuna-13B-1.1-HF"
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.model = LlamaForCausalLM.from_pretrained(
            base_model, load_in_8bit=True,
            torch_dtype=torch.float16, device_map={"": device_id}
        )
        
        lora_config = LoraConfig(
            r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha, lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules, bias=lora_args.bias, task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        
        path = snapshot_download(repo_id=model_path)
        weight = torch.load(f"{path}/pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(weight)
        
        self.device = f"cuda:{device_id}"
        
    def generate(self, prompt, max_new_tokens=500, min_new_tokens=100, early_stopping=True, do_sample=True, top_k=8, temperature=0.75, **kwargs):
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, 
            early_stopping=early_stopping, do_sample=do_sample, top_k=top_k, temperature=temperature, **kwargs
        )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded
        
    

if __name__ == "__main__":
    
    model = FLACUNA("declare-lab/flacuna-13b-v1.0")
    
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: You are tasked to demonstrate your writing skills in professional or work settings for the following question.\n"
        "Can you help me write a speech for a graduation ceremony, inspiring and motivating the graduates to pursue their dreams and make a positive impact on the world?\n"
        "Output: ASSISTANT: "
    )
    
    decoded = model.generate(prompt)
    print (decoded)