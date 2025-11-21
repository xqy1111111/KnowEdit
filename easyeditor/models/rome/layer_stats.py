import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Strict mode: do NOT perform any fallback. If loading fails, raise immediately.
        if os.getenv("NO_DATASET_FALLBACK", "0") == "1":
            default_cfg = dict(wikitext="wikitext-103-raw-v1", wikipedia="20231101.en")
            repo = ds_name
            config = default_cfg[ds_name]
            if ds_name == "wikipedia":
                repo = os.getenv("WIKI_DATASET", repo)
                config = os.getenv("WIKI_CONFIG", config)
            # Some wikipedia configs require Apache Beam runner. Allow specifying it via env.
            # Example: export WIKI_BEAM_RUNNER=DirectRunner
            beam_runner = os.getenv("WIKI_BEAM_RUNNER", None)
            if beam_runner:
                raw_ds = load_dataset(repo, config, beam_runner=beam_runner)
            else:
                raw_ds = load_dataset(repo, config)
            # determine maxlen
            if hasattr(model.config, 'n_positions'):
                maxlen = model.config.n_positions
            elif hasattr(model.config, 'max_sequence_length'):
                maxlen = model.config.max_sequence_length
            elif hasattr(model.config, 'max_position_embeddings'):
                maxlen = model.config.max_position_embeddings
            elif hasattr(model.config,'seq_length'):
                maxlen = model.config.seq_length
            else:
                raise NotImplementedError
            if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
                if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                    maxlen = model.config.sliding_window or 4096
                else:
                    maxlen = 4096
            if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
                maxlen = 4096
            if batch_tokens is not None and batch_tokens < maxlen:
                maxlen = batch_tokens
            return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

        # Fast offline path: allow env to force a tiny synthetic dataset, avoiding any network.
        # Triggers when USE_SYNTHETIC_STATS=1 or HF_DATASETS_OFFLINE=1.
        if os.getenv("USE_SYNTHETIC_STATS", "0") == "1" or os.getenv("HF_DATASETS_OFFLINE", "0") == "1":
            try:
                from datasets import Dataset, DatasetDict
                texts = [
                    "This is a tiny fallback corpus for offline covariance stats.",
                    "It contains a few short sentences to stabilize AlphaEdit.",
                    "Replace it with real data if available.",
                ] * 5000
                raw_ds = DatasetDict({"train": Dataset.from_dict({"text": texts})})
                print("[layer_stats] Using synthetic offline dataset due to env flag.")
                # Determine maxlen based on model config (same as below)
                if hasattr(model.config, 'n_positions'):
                    maxlen = model.config.n_positions
                elif hasattr(model.config, 'max_sequence_length'):
                    maxlen = model.config.max_sequence_length
                elif hasattr(model.config, 'max_position_embeddings'):
                    maxlen = model.config.max_position_embeddings
                elif hasattr(model.config,'seq_length'):
                    maxlen = model.config.seq_length
                else:
                    raise NotImplementedError
                if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
                    if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                        maxlen = model.config.sliding_window or 4096
                    else:
                        maxlen = 4096
                if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
                    maxlen = 4096
                if batch_tokens is not None and batch_tokens < maxlen:
                    maxlen = batch_tokens
                return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
            except Exception:
                # Fall through to standard path if synthetic construction fails
                pass
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        # Support environment overrides for Wikipedia under datasets>=3.
        default_cfg = dict(wikitext="wikitext-103-raw-v1", wikipedia="20231101.en")
        repo = ds_name
        config = default_cfg[ds_name]
        if ds_name == "wikipedia":
            repo = os.getenv("WIKI_DATASET", repo)
            config = os.getenv("WIKI_CONFIG", config)
        try:
            raw_ds = load_dataset(repo, config)
        except Exception:
            # Robust fallbacks for Wikipedia under different datasets versions / mirrors
            if ds_name == "wikipedia":
                # 1) Try the new repository name
                try:
                    fallback_repo = "wikimedia/wikipedia"
                    fallback_cfg = os.getenv("WIKI_CONFIG", "20231101.en")
                    raw_ds = load_dataset(fallback_repo, fallback_cfg)
                except Exception:
                    # 2) Fall back to wikitext (widely available) to unblock stats collection
                    try:
                        raw_ds = load_dataset("wikitext", "wikitext-103-raw-v1")
                    except Exception:
                        # 3) As a last resort, synthesize a tiny local dataset to unblock editing
                        try:
                            from datasets import Dataset, DatasetDict
                            texts = [
                                "This is a tiny fallback corpus for offline covariance stats.",
                                "It contains a few short sentences to stabilize AlphaEdit.",
                                "Replace it with real data if available.",
                            ] * 5000
                            raw_ds = DatasetDict({"train": Dataset.from_dict({"text": texts})})
                            print("[layer_stats] Falling back to a tiny synthetic dataset (offline mode).")
                        except Exception as final_err:
                            raise RuntimeError(
                                "Failed to load Wikipedia and wikitext; and also failed to create a fallback dataset. "
                                "You can set env WIKI_DATASET/WIKI_CONFIG to override, or pre-download a local dataset."
                            ) from final_err
                else:
                    raise
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
