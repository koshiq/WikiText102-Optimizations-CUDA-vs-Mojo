"""Interactive chat-style demo for the Transformer variants.

Usage:
  # Interactive REPL (default)
  python chat_demo.py

  # Single-shot prompt and exit
  python chat_demo.py --prompt "the meaning of life" --model both --gen-len 20

Notes:
- Tokenization is whitespace-based and uses the `data/wikitext-2` vocabulary.
- Words not present in the vocabulary are replaced with the `<eos>` token.
"""
import argparse
import torch
import time
from data import Corpus
from transformer import TransformerModel
from maxGraph_transformer import TransformerModel as TransformerModelMAX


def words_from_ids(ids, idx2word):
    return ' '.join([idx2word[i] for i in ids])


def tokenize_prompt(prompt, dictionary):
    # simple whitespace tokenizer that maps unknown words to '<eos>' id
    words = prompt.strip().split()
    eos_id = dictionary.word2idx.get('<eos>')
    ids = []
    for w in words:
        if w in dictionary.word2idx:
            ids.append(dictionary.word2idx[w])
        else:
            # map unknown to <eos>
            ids.append(eos_id)
    return ids


def generate_continuation(model, src_ids, gen_len, idx2word, device, sampling='greedy', topk=5):
    model.eval()
    model.to(device)
    src = torch.tensor(src_ids, dtype=torch.int64, device=device).unsqueeze(1)  # (seq_len, 1)

    generated_ids = list(src_ids)

    with torch.no_grad():
        for _ in range(gen_len):
            out = model(src)  # (seq_len, batch=1, vocab)
            logits = out[-1, 0, :]
            if sampling == 'greedy':
                next_id = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, topk)
                topk_probs = topk_probs / topk_probs.sum()
                next_id = int(topk_idx[torch.multinomial(topk_probs, 1)].item())

            generated_ids.append(next_id)
            # append to src for next iteration
            src = torch.cat([src, torch.tensor([[next_id]], dtype=torch.int64, device=device)], dim=0)

    return generated_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['pytorch', 'max', 'both'], default='both')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--gen-len', type=int, default=20)
    parser.add_argument('--sampling', choices=['greedy', 'topk'], default='greedy')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--no-timing', action='store_true', help='Suppress timing/benchmark prints')
    parser.add_argument('--load-model', type=str, default=None, help='Path to a PyTorch state_dict (model.pt) to load into the PyTorch model before generation')
    parser.add_argument('--ninp', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=512)
    parser.add_argument('--nlayers', type=int, default=3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    corpus = Corpus('data/wikitext-2')
    ntoken = len(corpus.dictionary)

    print(f"Loaded vocabulary (size={ntoken}). Using device: {device}")

    # Build models
    pytorch_model = TransformerModel(ntoken, args.ninp, args.nhead, args.nhid, args.nlayers)
    max_model = TransformerModelMAX(ntoken, args.ninp, args.nhead, args.nhid, args.nlayers)

    # Optionally load weights into pytorch_model
    if args.load_model:
        try:
            sd = torch.load(args.load_model, map_location='cpu')
            # If it's a dict with 'model_state_dict' key, try to extract
            if isinstance(sd, dict) and 'model_state_dict' in sd:
                sd = sd['model_state_dict']
            pytorch_model.load_state_dict(sd)
            print(f"Loaded weights from {args.load_model} into PyTorch model")
        except Exception as e:
            print(f"Warning: failed to load model from {args.load_model}: {e}")

    idx2word = corpus.dictionary.idx2word

    def run_prompt(prompt_text):
        ids = tokenize_prompt(prompt_text, corpus.dictionary)
        print(f"Input tokens: {words_from_ids(ids, idx2word)}")

        if args.model in ('pytorch', 'both'):
            t0 = time.perf_counter()
            gen_ids = generate_continuation(pytorch_model, ids, args.gen_len, idx2word, device, args.sampling, args.topk)
            t1 = time.perf_counter()
            print('\n--- PyTorch model output ---')
            print(words_from_ids(gen_ids, idx2word))
            if not args.no_timing:
                print(f"(generated {args.gen_len} tokens in {(t1-t0)*1000:.2f} ms)")

        if args.model in ('max', 'both'):
            t0 = time.perf_counter()
            try:
                gen_ids = generate_continuation(max_model, ids, args.gen_len, idx2word, device, args.sampling, args.topk)
                t1 = time.perf_counter()
                print('\n--- MAX model output ---')
                print(words_from_ids(gen_ids, idx2word))
                if not args.no_timing:
                    print(f"(generated {args.gen_len} tokens in {(t1-t0)*1000:.2f} ms)")
            except Exception as e:
                print(f"MAX model generation failed: {e}")

    if args.prompt:
        run_prompt(args.prompt)
        return

    # Interactive REPL
    print("Enter a prompt (whitespace-tokenized). Unknown words map to <eos>. Ctrl-D to exit.")
    try:
        while True:
            prompt_text = input('> ')
            if not prompt_text.strip():
                continue
            run_prompt(prompt_text)
    except EOFError:
        print('\nExiting.')


if __name__ == '__main__':
    main()
