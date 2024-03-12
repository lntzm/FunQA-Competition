import json

with open('./video_llama/results/hc.json', 'r') as f:
    results_hc = json.load(f)
with open('./video_llama/results/m.json', 'r') as f:
    results_m = json.load(f)

for i, j in zip(results_hc, results_m):
    if 'M' in i['task']:
        i['output'] = j['output']

with open('./output/卡不够用了.json', 'w') as f:
    json.dump(results_hc, f)