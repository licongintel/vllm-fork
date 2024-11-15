# Run_Accuracy

## Setup

### Clone repositories
```
git clone https://github.com/HabanaAI/vllm-fork.git
git clone ssh://gerrit.habana-labs.com:29418/vllm-benchmarks
```

### Checkout vllm-fork branch
```
cd vllm-fork
git checkout schoi/habana_main_1112+hpu_ext_pr24
```

### Docker Run
- Do not forget to set `no_proxy=localhost`
- If `/software` is not mounted right on host machine, copy below folder to the machine you are running accuracy tests `/software/data/mlperf/mlperf-loadgen/latest`
```
docker run -it --rm  \
-e HABANA_VISIBLE_DEVICES=<> \
-e HABANA_VISIBLE_MODULES=<> \
-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
--cap-add=sys_nice \
--name=vllm-accuracy_test  \
-v <path to cloned repositories>:/git \
-v <data_path>:/root/.cache/huggingface/hub \
-v <data_path>:/data  \
-v /software:/software  \
-e no_proxy=localhost \
-e HUGGING_FACE_HUB_TOKEN=<h> \
<docker image>
```

### Installation


Setup Accuracy tests: you may need to edit `/software` paths in `setup.sh` if it isn't mounted properly as mentioned in previous step
```
cd /git/vllm-benchmarks/benchmarks/acc-mlperf-moe
bash setup.sh
pip install -r /git/vllm-benchmarks/models/mixtral-8x7b/requirements.txt
pip install /git/vllm-fork
```

## Run Accuracy

### 1 card

Run vllm_server
```
bash /git/vllm-fork/IBM_benchmark/accuracy_test/run_vllm_server.sh -i 2048 -o 1024 -m mistralai/Mixtral-8x7B-Instruct-v0.1 -st 80 -t 1 -b 68 --output_dir /git/vllm-fork/IBM_benchmark/accuracy_test/1_card_log
```

Wait for Server to Start

Run Accuracy test
```
cd /git/vllm-benchmarks/benchmarks/acc-mlperf-moe
rm -rf results_1_card
python main.py --max-num-threads 68 --output-log-dir results_1_card --dataset-path /git/vllm-fork/IBM_benchmark/accuracy_test/2024.06.06_mixtral_15k_v4.pkl
```

Run Evaluation
```
python eval.py --checkpoint-path /data/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1 \
--dataset-file /git/vllm-fork/IBM_benchmark/accuracy_test/2024.06.06_mixtral_15k_v4.pkl \
--mlperf-accuracy-file results_1_card/mlperf_log_accuracy.json
```

Sample output
```json
{'rouge1': 45.5866, 'rouge2': 23.383, 'rougeL': 30.5323, 'rougeLsum': 42.5677, 'gsm8k': 72.04, 'mbxp': 59.58, 'gen_len': 4257309, 'gen_num': 15000, 'gen_tok_len': 2189320, 'tokens_per_sample': 146.0, 'performance': 0, 'accuracy': 97.64}
```


### 2 card

Run vllm_server
```
bash /git/vllm-fork/IBM_benchmark/accuracy_test/run_vllm_server.sh  -i 2048 -o 1024 -m mistralai/Mixtral-8x7B-Instruct-v0.1 -st 4 -t 2 -b 210 --output_dir /git/vllm-fork/IBM_benchmark/accuracy_test/2_card_log
```

Wait for Server to Start

Run Accuracy test
```
cd /git/vllm-benchmarks/benchmarks/acc-mlperf-moe
rm -rf results_2_card
python main.py --max-num-threads 210 --output-log-dir results_2_card --dataset-path /git/vllm-fork/IBM_benchmark/accuracy_test/2024.06.06_mixtral_15k_v4.pkl
```

Run Evaluation
```
python eval.py --checkpoint-path /data/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1 \
--dataset-file /git/vllm-fork/IBM_benchmark/accuracy_test/2024.06.06_mixtral_15k_v4.pkl \
--mlperf-accuracy-file results_2_card/mlperf_log_accuracy.json
```

Sample output
```json
{'rouge1': 45.6132, 'rouge2': 23.4586, 'rougeL': 30.5282, 'rougeLsum': 42.5815, 'gsm8k': 71.7, 'mbxp': 60.02, 'gen_len': 4258818, 'gen_num': 15000, 'gen_tok_len': 2189580, 'tokens_per_sample': 146.0, 'performance': 0, 'accuracy': 97.18}
```