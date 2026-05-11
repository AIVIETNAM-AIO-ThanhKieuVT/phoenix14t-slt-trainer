
import os
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phoenix14TDataset(Dataset):
    """
    Dataset loader for SLRTP Phoenix14T data (.pt files).
    Format:
    {
        'video_id': {
            'poses_3d': Tensor(T, 178, 3),
            'gloss': str,
            'text': str,
            ...
        }
    }
    Output:
        - pose: (T, 534) flattened 3D keypoints
        - text_tokens: Tokenized German text
        - gloss: Raw gloss string
    """
    def __init__(
        self,
        data_path,
        split='train',
        max_seq_len=400,
        max_text_len=128,
        normalize=True,
        stats_path=None,
        aug_temporal=False,
        aug_speed_range=(0.9, 1.1)
    ):
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        self.split = split
        self.aug_temporal = aug_temporal and split == 'train'
        self.aug_speed_range = aug_speed_range
        
        # 1. Load Data
        if os.path.isdir(data_path):
            file_path = os.path.join(data_path, f"{split}.pt")
        else:
            file_path = data_path

        logger.info(f"Loading data from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        self.data = torch.load(file_path, weights_only=False)
        self.video_ids = list(self.data.keys())
        logger.info(f"Loaded {len(self.video_ids)} samples.")

        # 2. Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # 3. Load/Compute Statistics
        self.stats = None
        if self.normalize:
            if stats_path is None:
                # ✅ FIX Bug #5: Use abspath to handle relative directory correctly
                data_dir = os.path.dirname(os.path.abspath(file_path))
                stats_path = os.path.join(data_dir, "stats_3d.pt")
            
            if os.path.exists(stats_path):
                logger.info(f"Loading statistics from {stats_path}")
                self.stats = torch.load(stats_path, weights_only=False)
            else:
                # ✅ FIX Bug #4: Always compute stats (or warn heavily) if split != train
                # This ensures consistent normalization between train and dev/test
                if split == 'train':
                    logger.info("Computing stats on the fly for training set...")
                    self.stats = self._compute_stats()
                else:
                    logger.warning(f"Stats file not found at {stats_path}. "
                                   f"Computing on-the-fly for {split} split. "
                                   f"WARNING: This may cause Distribution Shift if train stats differ!")
                    self.stats = self._compute_stats()

    def _compute_stats(self):
        """Compute Global Mean and Std for (178*3) features"""
        all_poses = []
        for vid in self.video_ids:
            # (T, 178, 3)
            pose = self.data[vid]['poses_3d']
            # Flatten to (T, 534)
            if torch.is_tensor(pose):
                flat_pose = pose.reshape(pose.shape[0], -1)
            else:
                flat_pose = torch.tensor(pose).reshape(pose.shape[0], -1)
            all_poses.append(flat_pose)
        
        # Concatenate all frames: (Total_Frames, 534)
        all_poses = torch.cat(all_poses, dim=0)
        
        mean = all_poses.mean(dim=0)
        # ✅ FIX Bug #10: Use unbiased=False (standard DL convention)
        std = all_poses.std(dim=0, unbiased=False)
        
        # Prevent division by zero
        std[std < 1e-6] = 1.0
        
        return {'mean': mean, 'std': std}

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        sample = self.data[vid_id]
        
        # 1. Process Pose
        # Input: (T, 178, 3)
        raw_pose = sample['poses_3d']
        if not torch.is_tensor(raw_pose):
            raw_pose = torch.tensor(raw_pose).float()
        
        # Flatten: (T, 534)
        pose = raw_pose.reshape(raw_pose.shape[0], -1).float()
        
        # 🟢 V4: Temporal Augmentation
        if self.aug_temporal:
            # 1. Random Speed Scaling (Linear Interpolation)
            speed_scale = np.random.uniform(self.aug_speed_range[0], self.aug_speed_range[1])
            new_len = int(pose.shape[0] * speed_scale)
            if new_len > 10: # Safety check
                # pose: [T, 534] -> [new_len, 534]
                pose_np = pose.numpy()
                old_indices = np.linspace(0, pose.shape[0]-1, pose.shape[0])
                new_indices = np.linspace(0, pose.shape[0]-1, new_len)
                
                scaled_pose = np.zeros((new_len, pose.shape[1]), dtype=np.float32)
                for d in range(pose.shape[1]):
                    scaled_pose[:, d] = np.interp(new_indices, old_indices, pose_np[:, d])
                pose = torch.from_numpy(scaled_pose)

        # 🆕 Get masks if they exist, else compute them from pose
        # ✅ FIX Bug #11: Compute masks BEFORE normalization. 
        # Zeros (missing keypoints) become non-zero after normalization.
        hand_mask = sample.get('hand_mask')
        face_mask = sample.get('face_mask')

        if hand_mask is None:
            # Body: 0-23, R Hand: 24-86, L Hand: 87-149, Face: 150-533
            hand_part = pose[:, 24:150]
            hand_mask = (hand_part != 0).any(dim=-1).float().unsqueeze(-1) # [T, 1]
        else:
            # ✅ FIX: Ensure consistent shape [T, 1] when loaded from data
            if not torch.is_tensor(hand_mask):
                hand_mask = torch.as_tensor(hand_mask).float()
            if hand_mask.dim() == 1:
                hand_mask = hand_mask.unsqueeze(-1)
            
        if face_mask is None:
            # Body: 0-23, R Hand: 24-86, L Hand: 87-149, Face: 150-533
            face_part = pose[:, 150:534]
            face_mask = (face_part != 0).any(dim=-1).float().unsqueeze(-1) # [T, 1]
        else:
            # ✅ FIX: Ensure consistent shape [T, 1] when loaded from data
            if not torch.is_tensor(face_mask):
                face_mask = torch.as_tensor(face_mask).float()
            if face_mask.dim() == 1:
                face_mask = face_mask.unsqueeze(-1)

        # Normalize
        if self.normalize and self.stats is not None:
            # ✅ FIX Bug #2: Ensure stats are on the same device as pose
            pose = (pose - self.stats['mean'].to(pose.device)) / self.stats['std'].to(pose.device)
            
        # Crop/Pad logic: Random start if sequence is too long (Time Shift Augmentation)
        seq_len = pose.shape[0]
        if seq_len > self.max_seq_len:
            if self.aug_temporal:
                start_idx = np.random.randint(0, seq_len - self.max_seq_len + 1)
            else:
                start_idx = 0
            pose = pose[start_idx : start_idx + self.max_seq_len]
            if hand_mask is not None:
                hand_mask = hand_mask[start_idx : start_idx + self.max_seq_len]
            if face_mask is not None:
                face_mask = face_mask[start_idx : start_idx + self.max_seq_len]
            seq_len = self.max_seq_len

        # 2. Process Text
        text = sample.get('text', "")
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'video_id': vid_id,
            'pose': pose,             # (T, 534)
            'seq_length': seq_len,
            'text': text,
            'gloss': sample.get('gloss', ''),
            'text_tokens': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'hand_mask': hand_mask,
            'face_mask': face_mask
        }

def collate_fn(batch):
    """
    Collate function to pad poses and stack tensors.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    video_ids = [item['video_id'] for item in batch]
    texts = [item['text'] for item in batch]
    glosses = [item['gloss'] for item in batch]
    
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Pad poses
    poses_list = [item['pose'] for item in batch]
    # (B, Max_T, 534)
    poses_padded = torch.nn.utils.rnn.pad_sequence(
        poses_list, 
        batch_first=True, 
        padding_value=0.0
    ) 
    
    # Create Pose Mask (B, Max_T)
    max_len = poses_padded.shape[1]
    pose_mask = (torch.arange(max_len)[None, :] < seq_lengths[:, None]).bool()
    
    # Pad masks if they exist
    hand_masks = [item['hand_mask'] for item in batch if item['hand_mask'] is not None]
    face_masks = [item['face_mask'] for item in batch if item['face_mask'] is not None]
    
    hand_mask_padded = None
    if hand_masks:
        # Convert to float tensor and pad: (B, Max_T, 1)
        hand_mask_tensors = [m.float() if torch.is_tensor(m) else torch.tensor(m).float() for m in hand_masks]
        hand_mask_padded = torch.nn.utils.rnn.pad_sequence(hand_mask_tensors, batch_first=True, padding_value=0.0)

    face_mask_padded = None
    if face_masks:
        face_mask_tensors = [m.float() if torch.is_tensor(m) else torch.tensor(m).float() for m in face_masks]
        face_mask_padded = torch.nn.utils.rnn.pad_sequence(face_mask_tensors, batch_first=True, padding_value=0.0)

    return {
        'video_ids': video_ids,
        'poses': poses_padded,         # (B, T, 534)
        'pose_mask': pose_mask, 
        'seq_lengths': seq_lengths,
        'text_tokens': text_tokens,
        'attention_mask': attention_masks,
        'texts': texts,
        'glosses': glosses,
        'hand_mask': hand_mask_padded, # 🆕 (B, T, 1)
        'face_mask': face_mask_padded  # 🆕 (B, T, 1)
    }

if __name__ == "__main__":
    # Test script and compute stats
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/Users/kieuvo/Learn/Research/SL/Implement/SignFML/slp-vtk/SignFML_DataExtraction_Phoenix14T/data")
    args = parser.parse_args()
    
    # 1. Compute Stats from Train
    logger.info("--- Computing Statistics form Train Set ---")
    train_path = os.path.join(args.data_dir, "train.pt")
    if os.path.exists(train_path):
        train_ds = Phoenix14TDataset(train_path, split='train', normalize=True)
        # Save stats
        stats_save_path = os.path.join(args.data_dir, "stats_3d.pt")
        torch.save(train_ds.stats, stats_save_path)
        logger.info(f"Saved stats to {stats_save_path}")
        
        # Verify stats
        mean = train_ds.stats['mean']
        std = train_ds.stats['std']
        logger.info(f"Mean shape: {mean.shape}, Max: {mean.max():.4f}, Min: {mean.min():.4f}")
        logger.info(f"Std shape: {std.shape}, Max: {std.max():.4f}, Min: {std.min():.4f}")
    
    # 2. Test Loading Dev
    logger.info("\n--- Testing Dev Set Loading ---")
    dev_path = os.path.join(args.data_dir, "dev.pt")
    if os.path.exists(dev_path):
        ds = Phoenix14TDataset(dev_path, split='dev', normalize=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        logger.info(f"Batch keys: {list(batch.keys())}")
        logger.info(f"Poses shape: {batch['poses'].shape}") # Should be (4, T, 534)
        logger.info(f"Mask shape: {batch['pose_mask'].shape}")