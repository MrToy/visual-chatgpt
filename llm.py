from typing import Any, List, Optional

import torch
from langchain.llms.base import LLM
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModel
from peft import  PeftModel, LoraConfig
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import re

ACTION_MAP = {
    "t2i": "Generate Image From User Input Text",
}


def find_all_images(input: str):
    return re.findall(r"(image/.+?\.png)", input)


class PromptParser:
    def __init__(self):
        self._history = []

    def parse_prompt(self, prompt: str):
        _, chat_history, input, agent_scratchpad = [i.strip() for i in prompt.split("==#==")]

        print(f"{chat_history = }")
        print(f"{input = }")

        return chat_history, input, agent_scratchpad

    def parse_output(self, output: str):
        m = re.search(r"<(\w+)>(.+)", output)
        if m:
            action = ACTION_MAP[m.group(1)]
            action_input = m.group(2)
            return action, action_input
        return None

PROMPT_FORMAT = """Instruction: {input_text}\nAnswer: """
class USER_LLM(LLM,BaseModel):
    model: Any
    tokenizer: Any
    prompt_parser: Any
    
    def __init__(self,model,tokenizer, **kwargs: Any):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_parser = PromptParser()

        
    @property
    def _llm_type(self) -> str:
        return "chatglm"
    
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # instruction = prompt
        print("prompt: "+prompt)
        chat_history, input, agent_scratchpad = self.prompt_parser.parse_prompt(prompt)
        if "Observation: image" in agent_scratchpad:
            images = "\n".join(find_all_images(agent_scratchpad))
            output = f" No\nAI: 这是你要的图片: {images}"
            return output
        
        with torch.no_grad():
            # feature = format_example({'instruction': f'{instruction}', 'output': '', 'input': ''})
            # input_text = feature['context']
            # input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
            format_input = PROMPT_FORMAT.format(input_text=input)
            ids = self.tokenizer.encode(format_input)
            input_ids = torch.LongTensor([ids])
            out = self.model.generate(input_ids=input_ids, max_length=512, do_sample=False, temperature=0)
            out_text = self.tokenizer.decode(out[0][len(ids) :])
            result = self.prompt_parser.parse_output(out_text)
            if result:
                action, action_input = result
                output = f" Yes\nAction: {action}\nAction Input: {action_input}"
            else:
                output = f" No\nAI: {out_text}"

            print("answer: "+output)
            return output
    
def GLM_LLM():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, "mymusise/chatglm-6b-alpaca-lora")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    llm = USER_LLM(model,tokenizer)
    return llm

def LLAMA_LLM():
    model = LlamaForCausalLM.from_pretrained(
        'decapoda-research/llama-7b-hf',
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
    model = PeftModel.from_pretrained(
        model,
        "tloen/alpaca-lora-7b",
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()  # seems to fix bugs for some users.
    model.eval()
    llm = USER_LLM(model,tokenizer)
    return llm
