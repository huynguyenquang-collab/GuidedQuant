from .helpers import dataloader
from tqdm import tqdm
import torch
from .helpers.utils import vprint, logprint, get_tokenizer_type, name_splitter, base_model_name_to_hf_repo_name
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..modules import AnyPrecisionForCausalLM
import os
import json
import pickle
import lm_eval
from datasets import load_dataset
from any_precision.quantization.torch_load import torch_load

current_dir = os.path.dirname(os.path.realpath(__file__))


def _truthy_env(name, default="0"):
    value = os.environ.get(name, default).lower()
    return value in {"1", "true", "yes", "on"}


def fake_pack(parent_path, verbose=True):
    # Load from non-packed parent model to simulate quantization
    # WARNING: This is for PPL research only, and should not be used for any other purpose
    import re
    logprint(verbose, f"Simulating Any-Precision model from non-packed parent model at {parent_path}")

    if os.path.isdir('./cache/fake_packed'):
        for file in os.listdir('./cache/fake_packed'):
            if parent_path.split("/")[-1] in file:
                logprint(verbose, f"Faked packed model already exists for {parent_path.split('/')[-1]}. Skipping...")
                return

    # Check if D&S quantization is used
    dns = parent_path.split("/")[-1].startswith("dns")

    fields = name_splitter(parent_path)
    # get the field wrapped in ()
    for field in fields:
        if field.startswith('(') and field.endswith(')'):
            base_model_name = field[1:-1]
            break
    else:
        raise ValueError(f"Could not find base model name in {parent_path}")
    original_model_repo = base_model_name_to_hf_repo_name(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_model_repo)

    logprint(verbose, f"Loading original model from {original_model_repo}")
    # Load the model from the original model repo
    model = AutoModelForCausalLM.from_pretrained(original_model_repo, torch_dtype=torch.float16,
                                                 trust_remote_code=True)

    logprint(verbose, f"Loading quantized weights from {parent_path}")
    # Load the qweights
    files = os.listdir(parent_path + '/weights')
    layer_count = len(files)  # this should suffice
    qweights = [None] * layer_count
    for file in tqdm(files, desc="Loading qweights", disable=not verbose):
        # filenames should be 'l0.pt'
        l = int(re.match(r'l(\d+).pt', file).group(1))
        qweights[l] = torch_load(parent_path + '/weights/' + file)

    logprint(verbose, f"Loading LUTs from {parent_path}")
    # get a list of directories in the model_path
    dirs = os.listdir(parent_path)
    dirs.remove('weights')
    if dns:
        dirs.remove('sparse')
    luts = {}
    # Only the LUT directories should remain
    for lut_dir in dirs:
        # example: lut_3
        bit = int(re.match(r'lut_(\d+)', lut_dir).group(1))
        for file in tqdm(os.listdir(parent_path + '/' + lut_dir), desc=f"Loading {bit}-bit LUTs",
                         disable=not verbose):
            if file.startswith('log_dict'):
                continue
            # example: l0.pt
            l = int(re.match(r'l(\d+).pt', file).group(1))
            if bit not in luts:
                luts[bit] = [None] * layer_count
            luts[bit][l] = torch_load(parent_path + '/' + lut_dir + '/' + file)

    # Load D&S sparse weights if they exist
    sparse_model_weights = []
    if dns:
        logprint(verbose, f"D&S quantization detected. Loading sparse weights...")
        for l in range(layer_count):
            sparse_weights = torch_load(parent_path + f'/sparse/l{l}.pt')
            sparse_model_weights.append(sparse_weights)

    logprint(verbose, f"Replacing qweights with centroids from LUTs...")

    max_bit = max(luts.keys())

    for bit in luts:
        state_dict = model.state_dict()
        for l in tqdm(range(layer_count), desc=f"Replacing qweights with {bit}-bit centroids", ):
            qweight = qweights[l]
            lut = luts[bit][l]
            for module_name in qweight:
                full_param_name_suffix = f".{l}.{module_name}.weight"
                matching_keys = [key for key in state_dict.keys() if key.endswith(full_param_name_suffix)]
                assert len(matching_keys) == 1, f"Expected 1 matching key, got {len(matching_keys)}"
                matching_key = matching_keys[0]

                module_qweight = qweight[module_name]
                module_lut = lut[module_name]
                module_weights = []
                for row_idx in range(module_qweight.shape[0]):
                    row_weights = []
                    for group_idx in range(module_qweight.shape[1]):
                        # fetch weights from the LUT
                        group_weights = module_lut[row_idx][group_idx][
                            module_qweight[row_idx][group_idx] >> (max_bit - bit)]
                        row_weights.append(torch.from_numpy(group_weights))
                    # join the group weights
                    row_weights = torch.cat(row_weights, dim=0)
                    module_weights.append(row_weights)
                module_weights = torch.stack(module_weights)
                # Add the sparse weights if they exist
                if dns:
                    sparse_weights = sparse_model_weights[l][module_name]
                    # get the indices of the sparse weights
                    sparse_indices = sparse_weights.indices()
                    # replace the weights with the sparse weights
                    module_weights[sparse_indices[0], sparse_indices[1]] = sparse_weights.values()
                state_dict[matching_key] = module_weights

        save_path = f'./cache/fake_packed/fake_anyprec-p{bit}-{parent_path.split("/")[-1]}'
        os.makedirs(save_path, exist_ok=True)
        torch.save(state_dict, save_path + '/pytorch_model.bin')
        tokenizer.save_pretrained(save_path)
        model.config.save_pretrained(save_path)
        logprint(verbose, f"{bit}-bit model saved to {save_path}")


@torch.no_grad()
def auto_model_load(model_path, device='cuda', dtype=torch.float16, verbose=True):
    """
    Args:
        model_path: path of the model to evaluate
        device: the device to use for evaluation, either 'cuda' or 'cpu'
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress

    Returns:
        (tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    if any(os.path.basename(model_path).startswith(prefix) for prefix in ["anyprec-", "layerwise-", "blockwise-", "full-"]):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AnyPrecisionForCausalLM.from_quantized(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        forced_device = os.environ.get("GUIDEDQUANT_CUDA_DEVICE")
        device_map = {"": f"cuda:{forced_device}"} if forced_device is not None else "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map,
                                                     trust_remote_code=True)

    logprint(verbose, f"{model.__class__.__name__} model loaded to device: {model.device}")

    tokenizer_type = get_tokenizer_type(model_path)

    if tokenizer_type is None:
        logprint(verbose, f"Unknown tokenizer type for {model_path}. Cannot use cached input tokens.")

    return tokenizer_type, tokenizer, model


def _get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _load_wikitext2_stream(cache_dir, seed, verbose):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wikitext2_test_seed{seed}.pkl")
    if os.path.exists(cache_file):
        logprint(verbose, f"Loading WikiText-2 stream from cache: {cache_file}")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    logprint(verbose, "Loading WikiText-2 test stream...")
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        token=_get_hf_token(),
    )
    full_text = "\n".join([text for text in dataset["text"] if text])
    texts = [full_text]

    logprint(verbose, f"Caching WikiText-2 stream to {cache_file}")
    with open(cache_file, "wb") as handle:
        pickle.dump(texts, handle)
    return texts


def _load_c4_validation_stream(n_samples, cache_dir, seed, verbose):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"c4_validation_n{n_samples}_seed{seed}.pkl")
    if os.path.exists(cache_file):
        logprint(verbose, f"Loading C4 stream from cache: {cache_file}")
        with open(cache_file, "rb") as handle:
            return pickle.load(handle)

    logprint(verbose, f"Loading C4 validation stream ({n_samples} documents)...")
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        token=_get_hf_token(),
    )
    texts = []
    for item in tqdm(dataset, total=n_samples, desc="Collecting C4", disable=not verbose):
        if len(texts) >= n_samples:
            break
        if len(item["text"].strip()) > 500:
            texts.append(item["text"])

    full_text = "\n\n".join(texts)
    joined_texts = [full_text]
    logprint(verbose, f"Caching C4 stream to {cache_file}")
    with open(cache_file, "wb") as handle:
        pickle.dump(joined_texts, handle)
    return joined_texts


def _load_ppl_texts(testcase_name, eval_samples, cache_dir, seed, verbose):
    if "wikitext2" in testcase_name.lower():
        return _load_wikitext2_stream(cache_dir, seed, verbose)
    if "c4" in testcase_name.lower():
        return _load_c4_validation_stream(eval_samples, cache_dir, seed, verbose)
    raise ValueError(f"Sliding-window PPL supports wikitext2 and c4, got {testcase_name}")


@torch.no_grad()
def _evaluate_sliding_window_ppl(model, tokenizer, texts, max_length, stride, verbose=True):
    device = _get_model_device(model)
    nlls = []
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encodings.input_ids

        if tokenizer.bos_token_id is not None:
            if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                input_ids = torch.cat([bos_tensor, input_ids], dim=1)

        if input_ids.size(1) > max_length * 200:
            input_ids = input_ids[:, : max_length * 200]

        input_ids = input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue

        window_range = list(range(0, seq_len, stride))
        logprint(verbose, f"Processing {seq_len:,} tokens in {len(window_range):,} windows...")

        prev_end_loc = 0
        pbar = tqdm(window_range, desc="Windows", unit="win", leave=False, disable=not verbose)
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_chunk = input_ids[:, begin_loc:end_loc]
            target_chunk = input_chunk.clone()
            if begin_loc > 0:
                target_chunk[:, :-trg_len] = -100

            if target_chunk.size(1) == 0:
                break

            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            current_nll = torch.stack(nlls).sum()
            current_ppl = torch.exp(current_nll / (total_tokens + prev_end_loc)).item()
            pbar.set_postfix({"PPL": f"{current_ppl:.4f}", "tokens": f"{total_tokens + prev_end_loc:,}"})

            if end_loc == seq_len:
                break

        total_tokens += seq_len

    if not nlls:
        return None

    total_nll = torch.stack(nlls).sum()
    perplexity = torch.exp(total_nll / total_tokens).item()
    return {"perplexity": perplexity, "total_tokens": total_tokens}


@torch.no_grad()
def evaluate_ppl(
        model,
        tokenizer,
        testcases,
        verbose=True,
        chunk_size=2048,
        tokenizer_type=None,
        stride=512,
        eval_samples=2000,
        cache_dir="./dataset_cache",
        seed=42,
        return_details=False):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    The perplexity scores are calculated with the same sliding-window protocol
    used by NonUQuantFix/RBVTSlidingWindowEvaluator: WikiText-2 and C4 are
    tokenized as continuous streams, evaluated with max_length=chunk_size and
    stride=stride, and capped at max_length * 200 tokens per stream.
    """

    if isinstance(model, AnyPrecisionForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)
            if _truthy_env("GUIDEDQUANT_CACHE_DEQUANT"):
                logprint(verbose, f"Caching {bit}-bit dense weights for fast fallback eval...")
                model.cache_dequantized_weights(bit)

        for testcase_name in testcases:
            vprint(verbose, f"---------------------- {testcase_name} ----------------------")

            texts = _load_ppl_texts(testcase_name, eval_samples, cache_dir, seed, verbose)
            logprint(verbose, "Calculating sliding-window perplexity...")

            dataset_result = _evaluate_sliding_window_ppl(
                model,
                tokenizer,
                texts,
                max_length=chunk_size,
                stride=stride,
                verbose=verbose,
            )
            if dataset_result is None:
                logprint(verbose, "Perplexity failed: no usable tokens")
                continue

            logprint(
                verbose,
                f"Perplexity: {dataset_result['perplexity']} | tokens={dataset_result['total_tokens']:,}",
            )

            key = f"{testcase_name}:{bit}-bit"
            results[key] = dataset_result if return_details else dataset_result["perplexity"]

        if not is_anyprec:
            break

    return results


def log_current_memory_usage(verbose=True, tag=''):
    if verbose:
        logprint(verbose, f"[{tag}] Current memory usage: {torch.cuda.memory_allocated() / (1 << 30)}GiB")

@torch.no_grad()
def evaluate_ppl_layerwise(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Forwards the input tokens layer by layer to reduce memory usage.
    """

    if isinstance(model, AnyPrecisionForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    dev = 'cuda'
    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    logprint(verbose, "Evaluating perplexity via layer-wise forward pass...")

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)
            child_model = model.model
        else:
            child_model = model

        for testcase_name in testcases:
            vprint(verbose, f"---------------------- {testcase_name} ----------------------")

            input_tokens = _load_input_tokens(tokenizer_type, testcase_name, tokenizer, chunk_size, verbose)

            input_tokens.to(dev)

            logprint(verbose, "Calculating perplexity...")

            seq_len = input_tokens.input_ids.size(1)
            nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

            input_ids = input_tokens.input_ids[:, :nsamples * chunk_size].view(nsamples, chunk_size).to(dev)

            batch_size = 1
            input_ids = [input_ids[i: i + batch_size] for i in range(0, nsamples, batch_size)]
            nbatches = len(input_ids)


            log_current_memory_usage(verbose, 'here0')
            inps = [0] * nbatches
            cache = {"i": 0, "attention_mask": None}

            class Catcher(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module

                def forward(self, inp, **kwargs):
                    inps[cache["i"]] = inp.cpu()
                    cache["i"] += 1
                    cache["attention_mask"] = kwargs["attention_mask"]
                    cache["position_ids"] = kwargs["position_ids"]
                    raise ValueError

            layers = child_model.model.layers
            child_model.model.embed_tokens = child_model.model.embed_tokens.to(dev)
            child_model.model.rotary_emb = child_model.model.rotary_emb.to(dev)

            layers[0] = layers[0].to(dev)

            layers[0] = Catcher(layers[0])

            for i in range(nbatches):
                batch = input_ids[i]
                try:
                    child_model(batch)
                except ValueError:
                    pass
                # log_current_memory_usage(verbose, 'here1')
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()

            child_model.model.embed_tokens = child_model.model.embed_tokens.cpu()
            position_ids = cache["position_ids"]

            torch.cuda.empty_cache()
            outs = [0] * nbatches
            attention_mask = cache["attention_mask"]

            for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
                layer = layers[i].to(dev)

                log_current_memory_usage(verbose, 'here2')
                for j in range(nbatches):
                    outs[j] = layer(
                        inps[j].to(dev),
                        attention_mask=attention_mask,
                        #  defined.
                        position_ids=position_ids,
                    )[0].cpu()
                layers[i] = layer.cpu()
                del layer
                log_current_memory_usage(verbose, 'here3')
                torch.cuda.empty_cache()
                log_current_memory_usage(verbose, 'here4')
                inps, outs = outs, inps

            if child_model.model.norm is not None:
                child_model.model.norm = child_model.model.norm.to(dev)

            child_model.lm_head = child_model.lm_head.to(dev)
            nlls = []
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            for i in range(nbatches):
                hidden_states = inps[i].to(dev)
                if child_model.model.norm is not None:
                    hidden_states = child_model.model.norm(hidden_states)
                lm_logits = child_model.lm_head(hidden_states)
                shift_logits = lm_logits[:, :-1, :]
                shift_labels = input_ids[i][:, 1:]
                loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
                neg_log_likelihood = loss.float().mean(dim=1)
                nlls.append(neg_log_likelihood)
            nlls_tensor = torch.cat(nlls)
            ppl = torch.exp(nlls_tensor.mean())

            logprint(verbose, f"Perplexity: {ppl.item()}")

            results[f"{testcase_name}:{bit}-bit"] = ppl.item()

        if not is_anyprec:
            break

    return results


@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, verbose=True):
    """ Run lm-eval on the given model and tasks and return the results.

    Receives an already initialized hf model, and a list of task names.
    """
    if isinstance(model, AnyPrecisionForCausalLM):
        is_anyprec = True
    else:
        is_anyprec = False

    model.eval()

    results = {}

    supported_bits = model.precisions if is_anyprec else [None]

    for bit in supported_bits:
        if is_anyprec:
            logprint(verbose, f"<<<< Setting model precision to {bit}-bit... >>>>")
            model.set_precision(bit)

        model_lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
        eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks)

        if verbose:
            logprint(verbose, json.dumps(eval_results['results'], indent=4))

        for task in tasks:
            results[f"{task}:{bit}-bit"] = eval_results['results'][task]

        if not is_anyprec:
            break

    return results


def _load_input_tokens(tokenizer_type, testcase_name, tokenizer, chunk_size, verbose):
    """ Load input tokens from cache if available, otherwise load from dataloader and save to cache. """
    input_tokens_cache_path = f"{current_dir}/input_tokens_cache/dataloader-{tokenizer_type}-{testcase_name}-test-ctx{chunk_size}.pt"
    if tokenizer_type and os.path.exists(input_tokens_cache_path):
        logprint(verbose, f"Loading cached input tokens from {input_tokens_cache_path}...")
        input_tokens = torch_load(input_tokens_cache_path)
    elif 'c4' in testcase_name:
        logprint(verbose, "Loading test set...")
        assert tokenizer is not None, "Tokenizer is required for C4-new"
        assert chunk_size is not None, "Chunk size is required for C4-new"

        input_tokens = dataloader.get_loaders(testcase_name, tokenizer, chunk_size)
    else:
        logprint(verbose, "Loading test set...")

        raw_text = dataloader.get_loaders(testcase_name)

        logprint(verbose, "Tokenizing test set...")

        input_tokens = tokenizer(raw_text, return_tensors='pt')

    # save input_tokens to cache
    if tokenizer_type and not os.path.exists(input_tokens_cache_path):
        logprint(verbose, f"Caching input tokens to {input_tokens_cache_path}...")
        # we must create the directory if it doesn't exist
        os.makedirs(os.path.dirname(input_tokens_cache_path), exist_ok=True)
        torch.save(input_tokens, input_tokens_cache_path)

    return input_tokens
