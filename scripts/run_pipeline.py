#!/usr/bin/env python3
"""Single entry point for running this repo's pipelines in their correct
dependency order. This wraps the individual scripts under scripts/attacks/
and scripts/defenses/ rather than replacing them -- each one stays runnable
and testable on its own; this just encodes the order so nobody has to know
it by heart or reverse-engineer it from docstrings.

Per-script tunable hyperparameters (steps, lr, n, epochs, ...) are NOT
exposed here -- run that one script directly if you need to override one.
This orchestrator only forwards the identity/path args every stage needs
(--data-root, --vq16-ckpt, ...) and encodes ordering.

List stages:
    python scripts/run_pipeline.py --list-stages

Run a stage:
    python scripts/run_pipeline.py --stage attacks-kermany \
        --data-root /path/to/chest_xray --llamagen-repo third_party/LlamaGen \
        --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt --meditok-repo third_party/meditok \
        --unitok-repo third_party/unitok --tokenizer-ckpt-dir third_party/ckpt \
        --fsq-repo third_party/FSQ-pytorch --fsq-ckpt third_party/ckpt/fsq_ckpt_16k.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt

    python scripts/run_pipeline.py --stage vq16-defenses-kermany \
        --data-root /path/to/chest_xray --llamagen-repo third_party/LlamaGen \
        --vq16-ckpt third_party/ckpt/vq_ds16_c2i.pt \
        --densenet-ckpt ckpts/classifiers/densenet121_kermany.pt \
        --probe-ckpt ckpts/linear_probes/probe_vq16_kermany.pt

If a step fails, fix the underlying issue and rerun the same --stage --
every script here is independently resumable (skips work whose output
already exists on disk), so restarting a stage does not redo finished steps.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def s(*parts):
    return os.path.join(ROOT, *parts)


# stage name -> ordered list of (script_path, [orchestrator arg names to forward])
STAGES = {
    "attacks-kermany": [
        (s("attacks", "run_vq16_kermany.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("attacks", "run_meditok_unitok_kermany.py"),
         ["data_root", "meditok_repo", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("attacks", "run_fsq_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("attacks", "build_summary_tables.py"), ["out_dir"]),
    ],
    "attacks-aptos": [
        (s("attacks", "run_vq16_aptos.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("attacks", "run_meditok_unitok_aptos.py"),
         ["data_root", "meditok_repo", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("attacks", "run_fsq_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("attacks", "build_summary_tables.py"), ["out_dir"]),
    ],
    "attacks-isic": [
        (s("attacks", "run_vq16_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("attacks", "run_meditok_unitok_isic.py"),
         ["data_root", "meditok_repo", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("attacks", "run_fsq_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("attacks", "build_summary_tables.py"), ["out_dir"]),
    ],
    "vq16-defenses-kermany": [
        # advft and decoder-realism are mutually independent (disjoint frozen/trainable
        # params); sweep_sqs only needs the undefended tokenizer -- all three could run in
        # parallel, but this pipeline runs them sequentially for simplicity.
        (s("defenses", "vq16_kermany", "run_advft_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "run_decoder_realism_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "sweep_sqs_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        # merge needs both advft and decoder-realism checkpoints to already exist
        (s("defenses", "vq16_kermany", "merge_checkpoints_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "significance_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "plot_sqs_sweeps_kermany.py"), ["out_dir"]),
        # everything below needs the SQS winner (from sweep_sqs) and/or the merged checkpoint
        (s("defenses", "vq16_kermany", "bpda_sqs_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "bpda_alone_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "geo_adaptive_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "cw_l2_heldout_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        (s("defenses", "vq16_kermany", "certify_sqs_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt"]),
        # the final table needs everything above to have finished
        (s("defenses", "vq16_kermany", "defense_eval_table_kermany.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "densenet_ckpt", "probe_ckpt"]),
    ],
    "vq16-defenses-aptos": [
        (s("defenses", "vq16_aptos", "run_advft_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "run_decoder_realism_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "sweep_sqs_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "merge_checkpoints_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "significance_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "plot_sqs_sweeps_aptos.py"), ["out_dir"]),
        (s("defenses", "vq16_aptos", "bpda_sqs_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "bpda_alone_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "geo_adaptive_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "cw_l2_heldout_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "certify_sqs_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_aptos", "defense_eval_table_aptos.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt", "probe_ckpt"]),
    ],
    "vq16-defenses-isic": [
        (s("defenses", "vq16_isic", "run_advft_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "run_decoder_realism_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "sweep_sqs_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "merge_checkpoints_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "significance_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "plot_sqs_sweeps_isic.py"), ["out_dir"]),
        (s("defenses", "vq16_isic", "bpda_sqs_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "bpda_alone_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "geo_adaptive_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "cw_l2_heldout_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "certify_sqs_isic.py"), ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt"]),
        (s("defenses", "vq16_isic", "defense_eval_table_isic.py"),
         ["data_root", "llamagen_repo", "vq16_ckpt", "effnet_ckpt", "probe_ckpt"]),
    ],
    "meditok-defenses-kermany": [
        # advft and decoder-realism are mutually independent; sweep_sqs only needs the
        # undefended tokenizers -- all three could run in parallel, but this pipeline runs
        # them sequentially for simplicity. Each script runs meditok only (split from the combined meditok_unitok_kermany pipeline).
        (s("defenses", "meditok_kermany", "run_advft_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "run_decoder_realism_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "sweep_sqs_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        # merge needs both advft and decoder-realism checkpoints to already exist
        (s("defenses", "meditok_kermany", "merge_checkpoints_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "significance_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "plot_sqs_sweeps_kermany.py"), ["out_dir"]),
        # everything below needs the SQS winner (from sweep_sqs) and/or the merged checkpoint
        (s("defenses", "meditok_kermany", "bpda_sqs_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "bpda_alone_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "geo_adaptive_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "cw_l2_heldout_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "meditok_kermany", "certify_sqs_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        # the final table needs everything above to have finished
        (s("defenses", "meditok_kermany", "defense_eval_table_kermany.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "densenet_ckpt", "probe_ckpt_dir"]),
    ],
    "unitok-defenses-kermany": [
        # advft and decoder-realism are mutually independent; sweep_sqs only needs the
        # undefended tokenizers -- all three could run in parallel, but this pipeline runs
        # them sequentially for simplicity. Each script runs unitok only (split from the combined meditok_unitok_kermany pipeline).
        (s("defenses", "unitok_kermany", "run_advft_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "run_decoder_realism_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "sweep_sqs_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        # merge needs both advft and decoder-realism checkpoints to already exist
        (s("defenses", "unitok_kermany", "merge_checkpoints_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "significance_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "plot_sqs_sweeps_kermany.py"), ["out_dir"]),
        # everything below needs the SQS winner (from sweep_sqs) and/or the merged checkpoint
        (s("defenses", "unitok_kermany", "bpda_sqs_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "bpda_alone_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "geo_adaptive_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "cw_l2_heldout_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        (s("defenses", "unitok_kermany", "certify_sqs_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt"]),
        # the final table needs everything above to have finished
        (s("defenses", "unitok_kermany", "defense_eval_table_kermany.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "densenet_ckpt", "probe_ckpt_dir"]),
    ],
    "fsq-defenses-kermany": [
        (s("defenses", "fsq_kermany", "run_advft_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "run_decoder_realism_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "sweep_sqs_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "merge_checkpoints_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "significance_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "plot_sqs_sweeps_kermany.py"), ["out_dir"]),
        (s("defenses", "fsq_kermany", "bpda_sqs_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "bpda_alone_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "geo_adaptive_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "cw_l2_heldout_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "certify_sqs_kermany.py"), ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt"]),
        (s("defenses", "fsq_kermany", "defense_eval_table_kermany.py"),
         ["data_root", "fsq_repo", "fsq_ckpt", "densenet_ckpt", "probe_ckpt"]),
    ],
    "fsq-defenses-aptos": [
        (s("defenses", "fsq_aptos", "run_advft_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "run_decoder_realism_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "sweep_sqs_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "merge_checkpoints_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "significance_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "plot_sqs_sweeps_aptos.py"), ["out_dir"]),
        (s("defenses", "fsq_aptos", "bpda_sqs_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "bpda_alone_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "geo_adaptive_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "cw_l2_heldout_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "certify_sqs_aptos.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_aptos", "defense_eval_table_aptos.py"),
         ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt", "probe_ckpt"]),
    ],
    "meditok-defenses-aptos": [
        (s("defenses", "meditok_aptos", "run_advft_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "run_decoder_realism_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "sweep_sqs_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "merge_checkpoints_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "significance_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "plot_sqs_sweeps_aptos.py"), ["out_dir"]),
        (s("defenses", "meditok_aptos", "bpda_sqs_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "bpda_alone_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "geo_adaptive_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "cw_l2_heldout_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "certify_sqs_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_aptos", "defense_eval_table_aptos.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt", "probe_ckpt_dir"]),
    ],
    "unitok-defenses-aptos": [
        (s("defenses", "unitok_aptos", "run_advft_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "run_decoder_realism_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "sweep_sqs_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "merge_checkpoints_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "significance_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "plot_sqs_sweeps_aptos.py"), ["out_dir"]),
        (s("defenses", "unitok_aptos", "bpda_sqs_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "bpda_alone_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "geo_adaptive_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "cw_l2_heldout_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "certify_sqs_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_aptos", "defense_eval_table_aptos.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt", "probe_ckpt_dir"]),
    ],
    "fsq-defenses-isic": [
        (s("defenses", "fsq_isic", "run_advft_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "run_decoder_realism_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "sweep_sqs_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "merge_checkpoints_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "significance_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "plot_sqs_sweeps_isic.py"), ["out_dir"]),
        (s("defenses", "fsq_isic", "bpda_sqs_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "bpda_alone_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "geo_adaptive_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "cw_l2_heldout_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "certify_sqs_isic.py"), ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt"]),
        (s("defenses", "fsq_isic", "defense_eval_table_isic.py"),
         ["data_root", "fsq_repo", "fsq_ckpt", "effnet_ckpt", "probe_ckpt"]),
    ],
    "meditok-defenses-isic": [
        (s("defenses", "meditok_isic", "run_advft_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "run_decoder_realism_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "sweep_sqs_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "merge_checkpoints_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "significance_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "plot_sqs_sweeps_isic.py"), ["out_dir"]),
        (s("defenses", "meditok_isic", "bpda_sqs_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "bpda_alone_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "geo_adaptive_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "cw_l2_heldout_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "certify_sqs_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "meditok_isic", "defense_eval_table_isic.py"),
         ["data_root", "meditok_repo", "tokenizer_ckpt_dir", "effnet_ckpt", "probe_ckpt_dir"]),
    ],
    "unitok-defenses-isic": [
        (s("defenses", "unitok_isic", "run_advft_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "run_decoder_realism_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "sweep_sqs_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "merge_checkpoints_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "significance_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "plot_sqs_sweeps_isic.py"), ["out_dir"]),
        (s("defenses", "unitok_isic", "bpda_sqs_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "bpda_alone_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "geo_adaptive_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "cw_l2_heldout_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "certify_sqs_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt"]),
        (s("defenses", "unitok_isic", "defense_eval_table_isic.py"),
         ["data_root", "unitok_repo", "tokenizer_ckpt_dir", "effnet_ckpt", "probe_ckpt_dir"]),
    ],
}

# orchestrator arg name -> the CLI flag every script that accepts it actually uses
ARG_TO_FLAG = {
    "data_root": "--data-root",
    "llamagen_repo": "--llamagen-repo",
    "vq16_ckpt": "--vq16-ckpt",
    "densenet_ckpt": "--densenet-ckpt",
    "effnet_ckpt": "--effnet-ckpt",
    "meditok_repo": "--meditok-repo",
    "unitok_repo": "--unitok-repo",
    "tokenizer_ckpt_dir": "--tokenizer-ckpt-dir",
    "fsq_repo": "--fsq-repo",
    "fsq_ckpt": "--fsq-ckpt",
    "probe_ckpt": "--probe-ckpt",
    "probe_ckpt_dir": "--probe-ckpt-dir",
    "out_dir": "--out-dir",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), help="which pipeline stage to run")
    p.add_argument("--list-stages", action="store_true", help="print every stage and its script order, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    for name in ARG_TO_FLAG:
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_stages or not args.stage:
        for stage, steps in STAGES.items():
            print(f"{stage}:")
            for script_path, _ in steps:
                print(f"  {os.path.relpath(script_path, ROOT)}")
        if not args.stage:
            print("\npass --stage <name> to run one")
        return

    for script_path, arg_names in STAGES[args.stage]:
        cmd = [sys.executable, script_path]
        for name in arg_names:
            value = getattr(args, name)
            if value is not None:
                cmd += [ARG_TO_FLAG[name], value]
        print(f"\n=== {os.path.relpath(script_path, ROOT)} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n*** {os.path.relpath(script_path, ROOT)} exited with code {result.returncode} -- "
                  f"stopping the pipeline here. Fix the issue and rerun --stage {args.stage}; "
                  f"already-finished steps will be skipped (each script checks for existing output). ***")
            sys.exit(result.returncode)

    print(f"\nDONE. stage '{args.stage}' completed.")


if __name__ == "__main__":
    main()
