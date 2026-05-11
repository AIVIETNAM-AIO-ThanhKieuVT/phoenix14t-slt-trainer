#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from models.fml.latent_flow_matcher_v4_27Feb import LengthPredictor, LatentFlowMatcherV4

class LatentFlowMatcherV4BERT(LatentFlowMatcherV4):
    """
    V4 variant that uses mBERT (768-dim) instead of mBART-50 (1024-dim).
    Also allows overriding max_seq_len to 400 for structural compatibility.
    """
    def __init__(self, latent_dim=256, hidden_dim=512, max_seq_len=400, **kwargs):
        # Prevent double print from super().__init__
        kwargs['text_encoder_name'] = kwargs.get('text_encoder_name', 'bert-base-multilingual-cased')
        super().__init__(latent_dim=latent_dim, hidden_dim=hidden_dim, max_seq_len=max_seq_len, **kwargs)
        
        # Override mBART with mBERT
        print(f"🔄 Overriding V4 with mBERT encoder: {kwargs['text_encoder_name']}")
        self.text_encoder = BertModel.from_pretrained(kwargs['text_encoder_name'])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.hidden_size # 768
        self.text_proj = nn.Linear(text_dim, self.hidden_dim)
        self.length_predictor = LengthPredictor(input_dim=text_dim)
        print(f"  📐 Text encoder dim adjusted to: {text_dim} (mBERT)")

    def encode_text(self, text_tokens, attention_mask):
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=text_tokens, attention_mask=attention_mask)
        raw_features = outputs.last_hidden_state  # (B, L, 768)
        text_features = self.text_proj(raw_features)  # (B, L, 512)
        text_mask = attention_mask.bool()
        return text_features, text_mask, raw_features
