export CUDA_VISIBLE_DEVICES=0

mkdir output
mkdir video_llama/results
python demo_funqa_multi.py --cfg-path ./eval_configs/all_finetune.yaml --classes HC --output_file ./video_llama/results/hc.json
python demo_funqa_multi.py --cfg-path ./eval_configs/splitm_finetune.yaml --classes M --output_file ./video_llama/results/m.json
python postprocess.py