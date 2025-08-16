# SLM Project (Modular, Parallel Training)

This project builds a tokenizer, tokenizes your data (CSV or XLSX), creates padded batches, trains a small Transformer SLM with **DDP parallelism** (multi-GPU or CPU), and runs inference.

## Install
```bash
pip install -r requirements.txt
```

## Data
Place your file at `data/sample.xlsx` or `data/sample.csv` with columns:
- `Text`
- `Keyword`
- `classification`
A tiny toy file is included.

## 1) Train Tokenizer
```bash
python train_tokenizer.py --input_file data/sample.xlsx --out_dir artifacts/tokenizer --vocab_size 3000
```

## 2) Tokenize Dataset
```bash
python tokenize_dataset.py --input_file data/sample.xlsx --tokenizer_dir artifacts/tokenizer --out_dir artifacts/data --max_length 128
```

## 3) Create Batches (pads to max_length, stores pad id)
```bash
python create_batches.py --data_path artifacts/data/tokenized.pt --out_path artifacts/data/batches.pt
```

## 4) Train SLM (parallel)
- **Single GPU / CPU:** `--world_size 1`
- **Multi-GPU on one machine:** set `--world_size` to the number of GPUs (e.g., 2)
- **CPU-only parallel:** set `--world_size` to number of CPU workers (e.g., 4)
```bash
python train_slm.py --data_path artifacts/data/batches.pt --model_out artifacts/model/final_model.pt --world_size 2 --epochs 5 --batch_size 32
```

> Uses DistributedDataParallel with backend **NCCL** on GPU and **Gloo** on CPU.

## 5) Inference
```bash
python inference.py --model_path artifacts/model/final_model.pt   --tokenizer_dir artifacts/tokenizer   --label_map_path artifacts/data/label_map.json   --text "Payment failed due to timeout." --keyword "payment"
```

## Notes
- Tokenizer saved as `vocab.json` + `merges.txt` (ByteLevelBPE). Meta with `pad_token_id` in `tokenizer_meta.json`.
- All scripts support **.csv** and **.xlsx** input.
- The model is a compact Transformer with LM+CLS heads; training optimizes both (next-token + class). Adjust `--lm_weight` in `train_slm.py` if desired.
