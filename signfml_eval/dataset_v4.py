#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset V4 — mBART-50 Tokenizer Wrapper
========================================
Author: SignFML Research
Date: 27 Feb 2026

Minimal subclass of Phoenix14TDataset that swaps:
    BertTokenizer → MBart50TokenizerFast (SentencePiece, German)

Pose data pipeline, collate_fn, and batch keys are IDENTICAL to dataset.py.
This ensures zero changes needed in the training loop.
"""

import logging
from transformers import MBart50TokenizerFast, MBart50Tokenizer
from dataset import Phoenix14TDataset, collate_fn  # Re-export collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MBART_MODEL_ID = 'facebook/mbart-large-50-many-to-many-mmt'


class Phoenix14TDatasetV4(Phoenix14TDataset):
    """
    V4: Same as Phoenix14TDataset but uses mBART-50 tokenizer.
    
    Key differences:
    - Tokenizer: MBart50TokenizerFast (SentencePiece) instead of BertTokenizer (WordPiece)
    - Token IDs: Different vocab, but same output format (input_ids, attention_mask)
    - Vocab size: ~250K (mBART) vs ~120K (mBERT)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ===== V4 CHANGE: Replace tokenizer =====
        logger.info("🆕 V4: Swapping to mBART-50 tokenizer (SentencePiece, de_DE)...")
        self.tokenizer = self._load_mbart_tokenizer()
        logger.info(f"  ✅ mBART tokenizer loaded (vocab_size={self.tokenizer.vocab_size})")
        # ===== END V4 CHANGE =====

    def _load_mbart_tokenizer(self):
        """Load mBART tokenizer with retries for cache/backend issues."""
        try:
            return MBart50TokenizerFast.from_pretrained(
                MBART_MODEL_ID,
                src_lang='de_DE',
            )
        except Exception as fast_err:
            logger.warning(f"Fast mBART tokenizer failed: {fast_err}")
            logger.info("Retrying fast tokenizer with force_download=True (cache refresh).")
            try:
                return MBart50TokenizerFast.from_pretrained(
                    MBART_MODEL_ID,
                    src_lang='de_DE',
                    force_download=True,
                )
            except Exception as fast_retry_err:
                logger.warning(f"Fast tokenizer retry failed: {fast_retry_err}")

            logger.info("Falling back to slow mBART tokenizer (requires sentencepiece).")
            try:
                return MBart50Tokenizer.from_pretrained(
                    MBART_MODEL_ID,
                    src_lang='de_DE',
                )
            except Exception as slow_err:
                raise RuntimeError(
                    "Unable to load mBART tokenizer. Install missing deps with: "
                    "`pip install sentencepiece protobuf` and clear corrupted HF cache if needed."
                ) from slow_err
