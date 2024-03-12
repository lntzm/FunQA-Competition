import json

def to_vl(official_ann, phase, output_file):
    b = []
    c = []
    with open(official_ann, 'r') as f:
        a = json.load(f)
    for i in a:
        if i['task'] != 'H1' and i['task'] != 'M1' and i['task'] != 'C1':
            b.append(i)
    for i in b:
        if i['visual_input'][0] == 'H':
            name = '%s_humor/'%phase+i['visual_input']
        elif i['visual_input'][0] == 'C':
            name = '%s_creative/'%phase+i['visual_input']
        elif i['visual_input'][0] == 'M':
            name = '%s_magic/'%phase+i['visual_input']
        else:
            raise ValueError
        if i["output"][0] == ' ':
            output = i["output"][1:]
        else:
            output = i["output"]
        c.append({"video": name, "QA": [{"q": i["instruction"], "a": output}]})

    with open(output_file, 'w') as w:
        json.dump(c, w)
        
def split():
    phase = 'train'
    H = []
    C = []
    M = []
    with open('../../official_data/annotation_with_ID/funqa_train.json', 'r') as f:
        a = json.load(f)
    for i in a:
        if i["output"][0] == ' ':
            output = i["output"][1:]
        else:
            output = i["output"]
            
        if i['visual_input'][0] == 'H':
            name = '%s_humor/'%phase+i['visual_input']
            H.append({"video": name, "QA": [{"q": i["instruction"], "a": output}]})
        elif i['visual_input'][0] == 'C':
            name = '%s_creative/'%phase+i['visual_input']
            C.append({"video": name, "QA": [{"q": i["instruction"], "a": output}]})
        elif i['visual_input'][0] == 'M':
            name = '%s_magic/'%phase+i['visual_input']
            M.append({"video": name, "QA": [{"q": i["instruction"], "a": output}]})
        else:
            raise ValueError

    # with open("./train_H.json", 'w') as w:
    #     json.dump(H, w)
    #     print('len H: ', len(H))
    # with open("./train_C.json", 'w') as w:
    #     json.dump(C, w)
    #     print('len C: ', len(C))
    with open("./train_M.json", 'w') as w:
        json.dump(M, w)
        print('len M: ', len(M))
    
        
if __name__ == '__main__':
    official_ann = '../../official_data/annotation_with_ID/funqa_train.json'
    phase = 'train'

    # 首先将原官方训练集数据funqa_train.json转化为本方案支持的格式
    # 输出文件为train.json
    to_vl(official_ann, phase, output_file='./train.json')

    # 再将扩充后的训练集数据funqa_aug_train.json转化为本方案支持的格式
    # 输出文件为aug_train.json
    aug_ann = './aug/funqa_aug_train.json'
    to_vl(aug_ann, phase, output_file='./aug_train.json')

    # 最后将原训练集里M的数据单独提出来用作微调
    # 输出文件为train_M.json
    split()