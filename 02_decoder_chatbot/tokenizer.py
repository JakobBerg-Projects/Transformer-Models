from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from pathlib import Path
from tqdm import tqdm

def example_to_text(example):
    return f"{example['question']} {example['answer']}"

def train_tokenizer(config):
    # Load the training data and select a subset to train the tokenizer on
    dataset = load_from_disk(config.dataset)

    # Concatenate the question and answer strings 
    print("Combining strings...")
    train_texts = [example_to_text(example) for example in tqdm(dataset)]

    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))
    tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, strip_accents=True)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[config.sep_token, config.end_token, config.pad_token, config.unk_token],
        show_progress=True,
    )

    # Train the tokenizer
    tokenizer.train_from_iterator(train_texts, trainer=trainer)

    # Save the tokenizer
    tokenizer_path = Path(config.tokenizer_filename)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")

    return tokenizer

if __name__ == "__main__":
    from config import config
    from utils import print_config

    print_config(config)

    if not Path(config.tokenizer_filename).exists():
        tokenizer = train_tokenizer(config)
    else:
        print(f"Using existing tokenizer at {config.tokenizer_filename}")
        tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    # Simple sanity check of the tokenizer
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    input_question = "How many legs does a cat have?"
    input_answer = "Five, unless it has lost one."

    text = f"{input_question} {config.sep_token} {input_answer} {config.end_token}"

    print(f"Question string: {input_question}")
    print(f"Answer string: {input_answer}")
    print(f"Tokenizer input: {text}")
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"Decoded: {decoded}")


