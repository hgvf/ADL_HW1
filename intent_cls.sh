python ./intent/inference.py --intent2idx_path ./intent/intent2idx.json --vocab_path ./intent/intent_vocab.pkl --embedding_path ./intent/intent_embeddings.pt --ckpt_dir ./intent/intent.pt --data_dir "${1}" --output_path "${2}"

