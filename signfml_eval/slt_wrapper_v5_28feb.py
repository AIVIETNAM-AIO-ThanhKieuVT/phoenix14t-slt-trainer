#!/usr/bin/env python3
"""
SLT Inference Wrapper for SignFML Evaluation
Wraps Sign-IDD-SLT model (based on SignJoey) to translate poses to text.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from collections import namedtuple
from typing import Optional, Union

# In this repo layout `back_translation/` lives at the repo root, which is the
# parent of signfml_eval/. Add the repo root to sys.path so imports resolve.
current_dir = Path(__file__).resolve().parent              # signfml_eval/
REPO_ROOT = current_dir.parent                              # repo root
sys.path.insert(0, str(REPO_ROOT))

try:
    from back_translation.bt_model import SignModel
    from back_translation.back_translate import back_translate, make_back_translation_model, load_config
    SLT_AVAILABLE = True
except ImportError as e:
    SLT_AVAILABLE = False
    print(f"⚠️ back_translation import failed: {e}")

# Dummy object to mimic torchtext batch
MockTorchBatch = namedtuple("MockTorchBatch", ["sgn", "sequence", "signer"])

class SLTWrapper:
    """Wrapper for temp_slrtp_eval model"""
    
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if not SLT_AVAILABLE:
            raise ImportError("temp_slrtp_eval logic not available.")
        
        # 1. Robust Model Directory Lookup
        if model_dir is None:
            potential_dirs = [
                PROJECT_ROOT / "SignFML_DataExtraction_Phoenix14T" / "backTranslation_PHIX_model",
                Path("/content/slp-vtk/SignFML_DataExtraction_Phoenix14T/backTranslation_PHIX_model"),
            ]
            for d in potential_dirs:
                if d.exists():
                    model_dir = d
                    break
        
        print(f"Loading Challenge SLT model from: {model_dir}")
        self.model = make_back_translation_model(model_dir=model_dir)
        
        if device is not None and 'cuda' in str(device):
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            # Force CPU on Mac for SLT specifically to avoid MPS compatibility issues with the old bt_model
            self.device = torch.device('cpu')
            
        self.model.to(self.device).eval()
        print(f"   ✅ Challenge SLT Model initialized on {self.device}.")
        
    def poses_to_text(self, poses, beam_size=3, beam_alpha=-1):
        """
        Translate list of pose tensors -> List of strings
        Args:
            poses: List of [T, D] tensors or [B, T, D] tensor
        Note: Data is assumed to be 25fps. SLT model expects ~12.5fps 
              (skeleton_subsample=2 in config), so we downsample [::2].
        """
        # Ensure poses is in the format back_translate expects: a list of [T, K, D] (non-flattened)
        if torch.is_tensor(poses):
            if poses.dim() == 3:
                if poses.shape[-1] == 534:
                    # Flattened -> [B, T, 178, 3]
                    poses = poses.reshape(poses.shape[0], poses.shape[1], 178, 3)
                poses_list = [poses[i] for i in range(poses.shape[0])]
            elif poses.dim() == 2:
                # [T, 534] -> [1, T, 178, 3]
                if poses.shape[-1] == 534:
                    poses = poses.reshape(poses.shape[0], 178, 3)
                poses_list = [poses]
            else:
                poses_list = poses
        else:
            poses_list = poses
            
        # ✅ Ensure all elements are torch.Tensor, on correct device and correctly shaped [T, 178, 3]
        final_list = []
        for p in poses_list:
            if not torch.is_tensor(p):
                p = torch.from_numpy(p)
            p = p.to(self.device).float()
            
            if p.dim() == 2 and p.shape[-1] == 534:
                p = p.reshape(-1, 178, 3)
            elif p.dim() == 3 and p.shape[0] == 1 and p.shape[-1] == 534:
                p = p.reshape(-1, 178, 3)
            
            # ✅ FIX: Subsample [::2] to match SLT model expectation (~12.5fps)
            # SLT model was trained with skeleton_subsample=2 (config.yaml)
            # Data is at 25fps, model expects every 2nd frame
            p = p[::2]
                
            final_list.append(p)
        poses_list = final_list

        # challenge back_translate function
        self.model.beam_size = beam_size
        self.model.beam_alpha = beam_alpha
        
        return back_translate(model=self.model, poses=poses_list)
        
    def compute_bleu(self, predicted_poses, reference_texts, tokenize='13a'):
        """Computes BLEU-4"""
        try:
            import sacrebleu
        except ImportError:
            print("⚠️ sacrebleu not found.")
            return None
            
        # Translate
        hypotheses = self.poses_to_text(predicted_poses)
        
        # Compute BLEU
        bleu_scores = {}
        for n in [1, 2, 3, 4]:
            score = sacrebleu.corpus_bleu(
                hypotheses, [reference_texts], 
                tokenize=tokenize, 
                max_ngram_order=n
            )
            bleu_scores[f'bleu-{n}'] = score.score
            
        return bleu_scores

def test_slt_wrapper():
    print("Testing SLT Wrapper integration...")
    try:
        slt = SLTWrapper()
        print("Wrapper initialized.")
        
        # Test with dummy zeros [2, 50, 534]
        dummy_poses = torch.zeros(2, 50, 534)
        print("Translating dummy poses...")
        texts = slt.poses_to_text(dummy_poses)
        print(f"Resulting texts: {texts}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_slt_wrapper()
