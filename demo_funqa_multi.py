import argparse
import os
import random
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

import random


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='llama_v2', help="The type of LLM")
    parser.add_argument("--classes", type=str, default='HMC', help="which class to evaluate")
    parser.add_argument("--output_file", type=str, default='./video_llama/results/hc.json', help="path to save results")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def gather_instructions(anno_file):
    with open(anno_file, 'r') as f:
        annos = json.load(f)
    filtered_annos = []
    for anno in annos:
        if anno['task'] not in ["H1", "M1", "C1", "C5"]:
            filtered_annos.append(anno)
    
    gathered_annos = {}
    for anno in filtered_annos:
        union_key = tuple([anno["visual_input"], anno["task"]])
        if union_key not in gathered_annos:
            gathered_annos[union_key] = set([anno["instruction"]])
        else:
            gathered_annos[union_key].add(anno["instruction"])
    
    return gathered_annos

with open('./cleaned_data/funqa_comp_test_submittd_formatted.json', 'r') as f:
    submit_ans = json.load(f)

all_instructions = gather_instructions(
    "../official_data/annotation_with_ID/funqa_test.json"
)

max_len = {
    'H2': 150,
    'H3': 180,
    'H4': 40,
    'C2': 390,
    'C3': 310,
    'C4': 30,
    'M2': 180,
    'M3': 130
}

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

for it in tqdm(submit_ans):
    file_name = it['visual_input']
    if file_name[0] == 'H' and 'H' in args.classes:
        file_name = '../official_data/test/test_humor/'+file_name
    elif file_name[0] == 'M' and 'M' in args.classes:
        file_name = '../official_data/test/test_magic/'+file_name
    elif file_name[0] == 'C' and 'C' in args.classes:
        file_name = '../official_data/test/test_creative/'+file_name
    else:
        continue
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    user_message = it['instruction']
    img_list = []
    llm_message = chat.upload_video_without_audio(file_name, chat_state, img_list, 32)
    chat.ask(user_message, chat_state)
    
    llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
    print(llm_message)
    limited_length = max_len[it['task']]
    if len(llm_message) > limited_length + limited_length * 0.05:
        print(f"out of length: {len(llm_message)}/{limited_length}")
        answers = []
        union_key = tuple([it["visual_input"], it["task"]])
        answer = None
        for i in range(20):
            print(f"retrying {i+1} times")
            candidates = list(all_instructions[union_key])
            index = random.randint(0, len(candidates)-1)
            user_message = candidates[index]
            chat_state = conv_llava_llama_2.copy()
            chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            img_list = []
            llm_message = chat.upload_video_without_audio(file_name, chat_state, img_list, 32)
            chat.ask(user_message, chat_state)
            
            llm_message = chat.answer(conv=chat_state,
                                      img_list=img_list,
                                      num_beams=1,
                                      temperature=1,
                                      max_new_tokens=300,
                                      max_length=2000)[0]
            print(llm_message)
            if len(llm_message) > limited_length + limited_length * 0.05:
                print(f"out of length: {len(llm_message)}/{limited_length}")
                answers.append(llm_message)
            else:
                print("profit length")
                answer = llm_message
                break
        if answer is None:
            print("No perfect lengths")
            answer = min(answers, key=len)
    else:
        answer = llm_message
    it['output'] = answer


with open(args.output_file, 'w') as f:
    json.dump(submit_ans, f)