import os
os.environ['http_proxy'] = "http://127.0.0.1:15225"
os.environ['https_proxy'] = "http://127.0.0.1:15225"

import openai
openai.api_key = ''

import re
import json
import random
import time
import datetime
from tqdm import tqdm


H_message = "You will be given a pair of question and answer in the following format: [question][answer], which is related to a counterintuitive (humorous, creative, or magical) video. Based on the given question, your task is to give 5 answers to the question with another types. Remember do not change the question, keep your answer of the same meaning with the given answer, and do not add many scene details which are not mentioned in the given answer. Also, your answer should keep similar length with the given one."

C_message = "You will be given a pair of question and answer in the following format: [question][answer], which is related to a creative video. Based on the given question, your task is to give 3 answers to the question with another types. List your 3 answers with number labels. Remember do not change the question. Keep your answer with the same meaning as the given answer, do not add many details which are not mentioned in the given answer, do not hide many details which are mentioned in the given answer. Also, your answer should keep similar length with the given one, not too long neither too short."


def gather_annos(anno_file):
    with open(anno_file, 'r') as f:
        annos = json.load(f)
    filtered_annos = []
    for anno in annos:
        if anno['task'] not in ["H1", "M1", "C1", "H4", "C4", "C5"]:
            filtered_annos.append(anno)
    
    gathered_annos = {}
    for anno in filtered_annos:
        union_key = tuple([anno["visual_input"], anno["task"]])
        if union_key not in gathered_annos:
            gathered_annos[union_key] = [anno]
        else:
            gathered_annos[union_key].append(anno)
    
    return gathered_annos


def get_gpt_response(response_id, gpt_input, log_file, err_file):
    visual_input, task = response_id
    if task.startswith("C"):
        system_message = C_message
    else:
        system_message = H_message
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{
                    'role': 'system',
                    'content': system_message,
                }, {
                    'role': 'user',
                    'content': gpt_input,
                }],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            ans = response['choices'][0]['message']['content']
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f"\n----------{now}----------\n[ID]{response_id}\n[INPUT]{gpt_input}\n[ANS]{ans}\n")
            # time.sleep(17)
            return ans
        except Exception as e:
            print('[ERROR]', e)
            ans =  '#ERROR#'
            time.sleep(20)
    if err_file is not None:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(err_file, 'a') as f:
            f.write(f"\n----------{now}----------\n[ID]{response_id}\n[INPUT]{gpt_input}\n[ERROR]{e}\n")
    return ans


def post_process(ans):
    ans_list = ans.split('\n')
    processed_ans = []
    for sentence in ans_list:
        if sentence == '':
            continue
        match = re.findall(r'\d+\.\s', sentence)
        assert len(match) == 1
        processed_sentence = sentence.split(match[0])[-1]
        processed_ans.append(processed_sentence)
    return processed_ans


def read_gpt_log(log_file):
    with open(log_file, 'r') as f:
        file = f.read()
    contents = file.split('\n----------')[1:]
    records_dict = {}
    for content in contents:
        # content = content.split('----------\n')[-1]
        [content, ans] = content.split('[ANS]')
        ans = ans.rstrip()
        if '[INPUT]' in content:
            [content, input] = content.split('[INPUT]')
        [content, key] = content.split('[ID]')
        key = eval(key.rstrip())
        record = {
            "key": key,
            "ans": ans,
        }
        # use dict to replace the older record with the newer one.
        records_dict[key] = record
    # records = []
    # for key, value in records_dict.items():
    #     records.append(value)
    return records_dict


def generate(resume_gpt):
    root = "./"
    anno_file = "../../../official_data/annotation_with_ID/funqa_train.json"
    log_file = os.path.join(root, "./gpt_log/gpt.log")
    err_file = os.path.join(root, "./gpt_log/gpt.err")
    save_file = os.path.join(root, "aug_result.jsonl")
    unaffordable_keys = [
        ('C_KT_13_4746_4850.mp4', 'C3'), ('C_KT_13_8150_8242.mp4', 'C2')
    ]
    with open(save_file, 'w') as f:
        pass
    
    gathered_annos = gather_annos(anno_file)
    if resume_gpt and os.path.exists(log_file):
        records_dict = read_gpt_log(log_file)
    for key, value in tqdm(gathered_annos.items()):
        if len(value[0]["output"]) > 470:
            continue
        if key in unaffordable_keys:
            continue
        visual_input, task = key
        if resume_gpt and key in records_dict:
            ans = records_dict[key]["ans"]
            ans = post_process(ans)
        else:
            error_flag = False
            for count in range(3):
                try:
                    task_length = len(value)
                    index = random.randint(0, task_length - 1)
                    anno = value[index]
                    question = anno["instruction"]
                    answer = anno["output"]
                    gpt_input = f"[{question}][{answer}]"
                    ans = get_gpt_response(key, gpt_input, log_file, err_file)
                    print(ans)
                    ans = post_process(ans)
                    break
                except AssertionError:
                    print("wrong format, retrying")
                    if count == 2:
                        error_flag = True
                    continue
            if error_flag:
                continue
        save_content = {
            "visual_input": visual_input,
            "task": task,
            "ans": ans,
            # "ID": anno["ID"]
        }
        with open(save_file, 'a') as f:
            f.write(json.dumps(save_content))
            f.write("\n")
   


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def gather_QA(anno_file):
    with open(anno_file, 'r') as f:
        annos = json.load(f)
    filtered_annos = []
    for anno in annos:
        if anno['task'] not in ["H1", "M1", "C1"]:
            filtered_annos.append(anno)
    
    gathered_instructions = {}
    gathered_outputs = {}
    for anno in filtered_annos:
        union_key = tuple([anno["visual_input"], anno["task"]])
        if union_key not in gathered_instructions:
            gathered_instructions[union_key] = set([anno["instruction"]])
            gathered_outputs[union_key] = set([anno["output"]])
        else:
            gathered_instructions[union_key].add(anno["instruction"])
            gathered_outputs[union_key].add(anno["output"])

    
    return gathered_instructions, gathered_outputs


def merge():
    root = "./"
    extra_data_path = os.path.join(root, "aug_result.jsonl")
    extra_data = load_jsonl(extra_data_path)
    extra_data_dict = {}
    for data in extra_data:
        union_key = tuple([data["visual_input"], data["task"]])
        extra_data_dict[union_key] = data["ans"]
    anno_file = "../../../official_data/annotation_with_ID/funqa_train.json"
    anno_instrutions, anno_outputs = gather_QA(anno_file)
    outputs = {}
    for key, value in tqdm(anno_outputs.items()):
        if key not in extra_data_dict:
            outputs[key] = list(value)
        else:
            outputs[key] = list(value) + extra_data_dict[key]
    new_data = []
    count = 0
    for key, value in anno_instrutions.items():
        instructions = list(value)
        answers = outputs[key]
        for instruction in instructions:
            for answer in answers:
                generated_data = {
                    "instruction": instruction,
                    "visual_input": key[0],
                    "output": answer,
                    "task": key[1],
                    "ID": f"train_{count}"
                }
                count += 1
                new_data.append(generated_data)
    with open(os.path.join(root, "funqa_aug_train.json"), "w") as f:
        f.write(json.dumps(new_data, indent=4))


if __name__ == "__main__":
    # 首先执行generate()函数使用gpt对原训练集进行同义句改写
    # resume_gpt=True时读取gpt_log目录下的gpt生成日志，并继续生成
    # 输出文件为aug_result.jsonl，存放gpt改写的结果
    generate(resume_gpt=True)

    # 再执行merge()函数，将生成的aug_result.jsonl与原官方数据合并到一起
    # 输出的合并后的文件为funqa_aug_train.json
    merge()
