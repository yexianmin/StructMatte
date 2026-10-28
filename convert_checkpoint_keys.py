#!/usr/bin/env python3
"""Convert original HRNIM-MRFA (HCMCFE) checkpoint keys to StructMatte.

Only parameter/buffer *names* are changed; tensor values and shapes stay intact.
Supports raw state_dict, Detectron2 {'model': ...}, or {'state_dict': ...} files.

Examples:
  python convert_checkpoint_keys.py -i checkpoint/model_final.pth -o checkpoint/structmatte_final.pth
  python convert_checkpoint_keys.py -i old.pth -o new.pth --dry-run

Use --trust-checkpoint only for trusted legacy files containing unsupported
pickle objects; never do this with an untrusted checkpoint.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import re
import sys

import torch

VIEW_NAMES = {
    f"fea_ex{level}_{view}": f"pse.view{view}_level{level + 1}"
    for view in (1, 2, 4)
    for level in range(3)
}
CSF_NAMES = {
    f"fea_fuison_layer{level}": f"cgsf.csf.level{level}"
    for level in (1, 2, 3)
}
CSF_NAMES["fea_fuison_layer4"] = "cgsf.csf.projection"
SCALAR_NAMES = {
    "alpha": "cgsf.input_scale",
    "gama": "cgsf.output_scale",
    "beita": "cgsf.bias",
}
LEGACY_NAMES = {**VIEW_NAMES, **CSF_NAMES, **SCALAR_NAMES}
# Only match complete path segments, never substrings of unrelated keys.
BRANCH_PATTERN = re.compile(r"(^|\.)HCMCFE\.(?P<component>[^.]+)(?=\.|$)")


def rename_key(key: str) -> str:
    """Rename an original key; preserve `module.` and any other leading prefix."""
    def replace(match: re.Match[str]) -> str:
        component = match.group("component")
        if component not in LEGACY_NAMES:
            raise ValueError(f"Unrecognized parameter in original HCMCFE branch: {key}")
        return f'{match.group(1)}structure_enhancement.{LEGACY_NAMES[component]}'
    return BRANCH_PATTERN.sub(replace, key)


def looks_like_state_dict(obj: object) -> bool:
    return (isinstance(obj, collections.abc.Mapping)
            and bool(obj)
            and all(isinstance(k, str) for k in obj.keys())
            and all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in obj.values()))


def select_state_dict(checkpoint: object, model_key: str | None):
    if model_key:
        if not isinstance(checkpoint, collections.abc.Mapping) or model_key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain requested key: {model_key!r}")
        target = checkpoint[model_key]
        if not looks_like_state_dict(target):
            raise ValueError(f"{model_key!r} is not a tensor state_dict")
        return target, model_key
    if looks_like_state_dict(checkpoint):
        return checkpoint, None
    if isinstance(checkpoint, collections.abc.Mapping):
        for key in ("model", "state_dict", "model_state_dict", "ema_state_dict", "model_ema"):
            if key in checkpoint and looks_like_state_dict(checkpoint[key]):
                return checkpoint[key], key
    raise ValueError("Could not locate a tensor state_dict. Use --model-key if necessary.")


def convert_state_dict(state_dict: collections.abc.Mapping[str, torch.Tensor]):
    new_dict = collections.OrderedDict()
    changed = []
    for old_key, tensor in state_dict.items():
        new_key = rename_key(old_key)
        if new_key in new_dict:
            raise ValueError(f"Key collision: {old_key!r} maps to an existing key {new_key!r}")
        new_dict[new_key] = tensor
        if new_key != old_key:
            changed.append((old_key, new_key))
    # Torch state_dict stores module-version metadata indexed by module path.
    if hasattr(state_dict, "_metadata"):
        metadata = collections.OrderedDict()
        for key, value in state_dict._metadata.items():
            match = re.search(r"(^|\.)HCMCFE(?:\.(?P<component>[^.]+))?", key)
            if match:
                component = match.group("component")
                if component is None:
                    new = key[: match.start()] + match.group(1) + "structure_enhancement"
                elif component in LEGACY_NAMES:
                    path = f"structure_enhancement.{LEGACY_NAMES[component]}"
                    new = key[: match.start()] + match.group(1) + path + key[match.end():]
                else:
                    # Original up/downsampling nodes had no trainable parameters;
                    # their module metadata is obsolete in the renamed hierarchy.
                    continue
            else:
                new = key
            if new in metadata:
                raise ValueError(f"Metadata key collision after rename: {new}")
            metadata[new] = value
        new_dict._metadata = metadata
    return new_dict, changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Original .pth/.pt checkpoint")
    parser.add_argument("-o", "--output", type=Path, required=False, help="New checkpoint path")
    parser.add_argument("--model-key", help="Explicit key containing the model state_dict (e.g. 'model')")
    parser.add_argument("--dry-run", action="store_true", help="List key changes without writing")
    parser.add_argument("--model-only", action="store_true", help="Write only renamed model parameters; omit metadata, optimizer and scheduler")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Permit full Python pickle loading; TRUSTED checkpoints only")
    parser.add_argument("--allow-no-change", action="store_true", help="Allow a checkpoint with no legacy keys")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Input not found: {source}")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.output and source == args.output.expanduser().resolve():
        parser.error("Refusing to overwrite the original checkpoint")

    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=not args.trust_checkpoint)
    except Exception as exc:
        if not args.trust_checkpoint:
            print("Safe checkpoint loading failed. For your own trusted legacy weights,")
            print("repeat with --trust-checkpoint. Do NOT use it for unknown files.", file=sys.stderr)
        raise

    state_dict, selected_key = select_state_dict(checkpoint, args.model_key)
    new_dict, changes = convert_state_dict(state_dict)
    print(f"Input: {source}")
    print(f"Model state dict: {selected_key or '<root>'}")
    print(f"Total parameter/buffer keys: {len(state_dict)}")
    print(f"Renamed keys: {len(changes)}")
    for before, after in changes[:18]:
        print(f"  {before}  ->  {after}")
    if len(changes) > 18:
        print(f"  ... {len(changes) - 18} additional renames")
    if not changes and not args.allow_no_change:
        raise ValueError("No HCMCFE keys found. Is this already converted, or a backbone-only checkpoint?")
    if args.dry_run:
        print("Dry run: no file saved.")
        return 0
    assert args.output is not None
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.model_only or selected_key is None:
        result = new_dict
    else:
        # Copy top-level fields without modifying non-model metadata or optimizer states.
        result = checkpoint.copy() if hasattr(checkpoint, "copy") else dict(checkpoint)
        result[selected_key] = new_dict
    torch.save(result, destination)
    print(f"Saved: {destination}")
    print("Only model key names were changed; tensor data/dtypes/shapes were preserved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
