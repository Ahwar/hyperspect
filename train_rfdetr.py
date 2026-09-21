# disable FutureWarning
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore')
import argparse
import logging
import os
import yaml
from pathlib import Path
import torch

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback
    except ImportError:
        class Callback: pass  # Fallback just in case

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopFileCallback(Callback):
    """
    A callback to monitor for a 'stopfile.txt' at the end of each epoch.
    Matches the custom stopping logic from the original YOLOv5 train.py.
    """
    def __init__(self, stopfile_path="stopfile.txt"):
        super().__init__()
        self.stopfile_path = stopfile_path
        
    def on_train_epoch_end(self, trainer, pl_module):
        if os.path.exists(self.stopfile_path):
            with open(self.stopfile_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().lower() == "stop":
                        logger.info("Stop file contains 'stop', stopping training.")
                        trainer.should_stop = True
                        return
                    elif line.strip().isdigit():
                        stop_epoch = int(line.strip())
                        if trainer.current_epoch >= stop_epoch:
                            logger.info(f"Stop file contains epoch {stop_epoch}, stopping training at epoch {trainer.current_epoch}.")
                            trainer.should_stop = True
                            return


import numpy as np
from PIL import Image

class DualStreamWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.actual_dataset = base_dataset.dataset if isinstance(base_dataset, torch.utils.data.Subset) else base_dataset
        if not hasattr(self.actual_dataset, 'root'):
            raise ValueError("Dataset does not have 'root' attribute.")
        
        self.se_root = Path(self.actual_dataset.root)
        self.sa_root = Path(str(self.actual_dataset.root).replace('se_information', 'sa_information').replace('dual_information', 'sa_information'))
        self.dual_root = Path(str(self.actual_dataset.root).replace('se_information', 'dual_information').replace('sa_information', 'dual_information'))

    def __len__(self):
        return len(self.base_dataset)

    def _process_pil_pair(self, img_rgb_pil, img_ir_pil, idx):
        actual_idx = self.base_dataset.indices[idx] if isinstance(self.base_dataset, torch.utils.data.Subset) else idx
        image_id = self.actual_dataset.ids[actual_idx]
        annotations = self.actual_dataset._load_target(image_id)
        
        import random
        state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()
        
        target_rgb = {"image_id": image_id, "annotations": annotations}
        img_rgb, target_out = self.actual_dataset.prepare(img_rgb_pil, target_rgb)
        if self.actual_dataset._transforms is not None:
            img_rgb, target_out = self.actual_dataset._transforms(img_rgb, target_out)
            
        torch.set_rng_state(state)
        random.setstate(random_state)
        np.random.set_state(np_state)
        
        target_ir = {"image_id": image_id, "annotations": annotations}
        img_ir, _ = self.actual_dataset.prepare(img_ir_pil, target_ir)
        if self.actual_dataset._transforms is not None:
            img_ir, _ = self.actual_dataset._transforms(img_ir, _)
            
        img_dual = torch.cat([img_rgb, img_ir], dim=0)
        return img_dual, target_out

    def __getitem__(self, idx):
        import random
        
        actual_idx = self.base_dataset.indices[idx] if isinstance(self.base_dataset, torch.utils.data.Subset) else idx
        image_id = self.actual_dataset.ids[actual_idx]
        file_name = self.actual_dataset.coco.loadImgs(image_id)[0]["file_name"]
        stem = Path(file_name).stem
        
        npy_path = self.dual_root / f"{stem}.npy"
        if npy_path.exists():
            dual_arr = np.load(npy_path)
            img_rgb_pil = Image.fromarray(dual_arr[:, :, :3])
            img_ir_pil = Image.fromarray(dual_arr[:, :, 3:])
            return self._process_pil_pair(img_rgb_pil, img_ir_pil, idx)
        
        state = torch.get_rng_state()
        random_state = random.getstate()
        np_state = np.random.get_state()
        
        orig_root = self.actual_dataset.root
        self.actual_dataset.root = self.se_root
        img_rgb, target = self.base_dataset[idx]
        
        torch.set_rng_state(state)
        random.setstate(random_state)
        np.random.set_state(np_state)
        
        self.actual_dataset.root = self.sa_root
        img_ir, _ = self.base_dataset[idx]
        self.actual_dataset.root = orig_root
        
        img_dual = torch.cat([img_rgb, img_ir], dim=0)
        return img_dual, target


def parse_args():
    default_data = './data/hsi/custom_hod_coco.yaml' if os.path.exists('./data/hsi/custom_hod_coco.yaml') else './data/hsi/custom_hod.yaml'
    parser = argparse.ArgumentParser(description="RF-DETR Training Script")
    parser.add_argument('--data', type=str, default=default_data, help='data.yaml path')
    parser.add_argument('--weights', type=str, default='rfdtr-small.pth', help='pretrained weights path')
    parser.add_argument('--model', type=str, default='small', choices=['small', 'large', 'xxlarge'], help='model size')
    parser.add_argument('--stream-mode', type=str, default='rgb', choices=['rgb', 'ir', 'stack', 'dual'], 
                        help='stream strategy: rgb (default 3ch), ir (3ch), stack (pseudo 3ch), dual (6ch)')
    parser.add_argument('--img-size', type=int, default=576, help='image size (must be divisible by 32)')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--project', default='project/rfdetr/train', help='save to project/name')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='enable gradient checkpointing')
    parser.add_argument('--wandb', action='store_true', help='use wandb logging')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='use tensorboard logging')
    return parser.parse_args()


def main():
    
    opt = parse_args()
    
    # 1. Parse data.yaml
    logger.info(f"Loading data config from {opt.data}")
    with open(opt.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    num_classes = int(data_dict.get('nc', 1))
    class_names = data_dict.get('names', [f'class_{i}' for i in range(num_classes)])
    
    # 2. Resolve dataset directory based on stream-mode
    train_rgb_path = Path(data_dict.get('train_rgb', ''))
    train_ir_path = Path(data_dict.get('train_ir', ''))
    
    num_channels = 3
    if opt.stream_mode == 'rgb':
        stream_path = train_rgb_path
    elif opt.stream_mode == 'ir':
        stream_path = train_ir_path
    elif opt.stream_mode == 'stack':
        # stack_information was pre-generated by convert_dataset_rfdetr.py
        # Each image is a 3-ch pseudo: [gray(SE), gray(SA), |gray(SE)-gray(SA)|]
        train_stack_raw = data_dict.get('train_stack', '')
        if not train_stack_raw:
            raise ValueError(
                "'train_stack' key missing from data YAML. "
                "Re-run convert_dataset_rfdetr.py to generate the stack_information folder."
            )
        stream_path = Path(train_stack_raw)
        logger.info("Stack mode: using pre-merged 3-ch pseudo images from stack_information/")
    elif opt.stream_mode == 'dual':
        train_dual_raw = data_dict.get('train_dual', '')
        if not train_dual_raw:
            train_dual_raw = data_dict.get('train_rgb', '')
        stream_path = Path(train_dual_raw)
        if not stream_path.exists() or any(stream_path.glob("*.npy")):
            # If stream_path points to dual_information containing .npy files,
            # build_dataset needs PNG images for initial COCO loading metadata/annotations,
            # so we fall back stream_path to train_rgb_path (se_information) for dataset building structure.
            stream_path = train_rgb_path
        num_channels = 6
        
        logger.info("Dual mode selected. Wrapping datasets to concatenate RGB and IR on the fly (supporting .npy files).")

        import rfdetr.datasets
        original_build = rfdetr.datasets.build_dataset
        def patched_build(*args, **kwargs):
            dataset = original_build(*args, **kwargs)
            return DualStreamWrapper(dataset)
        rfdetr.datasets.build_dataset = patched_build
    else:
        stream_path = train_rgb_path

    # Determine root directory:
    # If YOLO style: path/to/stream/images/train -> root is path/to/stream (parent of images)
    # If COCO style: path/to/stream/train/       -> root is path/to/stream (parent of train)
    if (stream_path.parent / "train" / "_annotations.coco.json").exists() or (stream_path.parent / "train").exists():
        dataset_dir = stream_path.parent
    elif (stream_path.parent.parent / "train" / "images").exists() or (stream_path.parent.parent / "data.yaml").exists():
        dataset_dir = stream_path.parent.parent
    else:
        # Fallback: check if stream_path itself is the 'train' folder or 'images/train'
        if stream_path.name == "train" and stream_path.parent.name == "images":
            dataset_dir = stream_path.parent.parent
        else:
            dataset_dir = stream_path.parent

    dataset_dir = str(dataset_dir.resolve() if hasattr(dataset_dir, "resolve") else Path(dataset_dir).resolve())
    logger.info(f"Using dataset directory: {dataset_dir} for stream-mode: {opt.stream_mode}")
    
    # 3. Setup output directory
    output_dir = Path(opt.project) / opt.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Initialize Model
    logger.info(f"Initializing RF-DETR '{opt.model}' model... (channels: {num_channels})")
    
    model_kwargs = {
        # 'pretrain_weights': opt.weights if num_channels == 3 else None, 
        'num_classes': num_classes,
        'resolution': opt.img_size,
        'num_channels': num_channels,
        'gradient_checkpointing': opt.gradient_checkpointing
    }
    
    if opt.model == 'small':
        from rfdetr import RFDETRSmall
        model = RFDETRSmall(**model_kwargs)
    elif opt.model == 'large':
        from rfdetr import RFDETRLarge
        model = RFDETRLarge(**model_kwargs)
    elif opt.model == 'xxlarge':
        from rfdetr import RFDETRXXLarge
        model = RFDETRXXLarge(**model_kwargs)
    else:
        from rfdetr import RFDETR
        model = RFDETR(**model_kwargs)

    # If num_channels != 3, patch Dinov2WithRegistersPatchEmbeddings.__init__ so that
    # BOTH the main model and the EMA copy are constructed with the right in_channels.
    # Patching forward (lazy expansion) breaks EMA because the EMA model is created
    # before the first forward pass, leaving it with a 3-ch projection.
    #
    # We also patch load_pretrain_weights to expand the projection weight in any
    # pretrained checkpoint (3-ch) to match the model's 6-ch projection before loading.
    if num_channels != 3:
        logger.info(f"Patching DINOv2 PatchEmbeddings for {num_channels} input channels...")
        import copy as _copy
        from rfdetr.models.backbone.dinov2_with_windowed_attn import Dinov2WithRegistersPatchEmbeddings as _PatchEmbCls
        from rfdetr.models import weights as _wmod

        _target_channels = num_channels

        # 1) Patch __init__ so every new instance (main model + EMA) gets num_channels in_channels
        _orig_patch_init = _PatchEmbCls.__init__

        def _multichannel_patch_init(self, config):
            config_copy = _copy.copy(config)
            config_copy.num_channels = _target_channels
            _orig_patch_init(self, config_copy)

        _PatchEmbCls.__init__ = _multichannel_patch_init

        # 2) Patch load_pretrain_weights to expand 3-ch projection weights in any
        #    pretrained checkpoint to _target_channels before calling load_state_dict.
        _orig_lpw = _wmod.load_pretrain_weights

        def _multichannel_lpw(nn_model, model_config):
            _orig_lsd = torch.nn.Module.load_state_dict

            def _expanded_lsd(self, state_dict, strict=True, **kwargs):
                expanded = {}
                for k, v in state_dict.items():
                    if ('patch_embeddings.projection.weight' in k
                            and v.ndim == 4
                            and v.shape[1] < _target_channels):
                        reps = _target_channels // v.shape[1]
                        expanded[k] = v.repeat(1, reps, 1, 1) / reps
                        logger.info(f"  Expanded checkpoint weight {k}: {tuple(v.shape)} -> {tuple(expanded[k].shape)}")
                    else:
                        expanded[k] = v
                return _orig_lsd(self, expanded, strict=strict, **kwargs)

            torch.nn.Module.load_state_dict = _expanded_lsd
            try:
                _orig_lpw(nn_model, model_config)
            finally:
                torch.nn.Module.load_state_dict = _orig_lsd

        _wmod.load_pretrain_weights = _multichannel_lpw


    # 5. Add custom callbacks
    # RFDETR maintains a callbacks dictionary matching lightning callback hooks
    if "trainer" not in model.callbacks:
        model.callbacks["trainer"] = []
    model.callbacks["trainer"].append(StopFileCallback())
    
    # 6. Set device/accelerator
    if opt.device == 'cpu':
        accelerator = 'cpu'
        devices = 1
    else:
        accelerator = 'gpu'
        devices = [int(x) for x in str(opt.device).split(',')] if ',' in str(opt.device) else int(opt.device)
        
    # 7. Start Training
    logger.info("Starting RF-DETR training...")
    try:
        
        model.train(
            dataset_dir=dataset_dir,
            epochs=opt.epochs,
            batch_size=opt.batch_size,
            num_workers=opt.workers,
            output_dir=str(output_dir),
            resume=opt.resume,
            accelerator='auto',
            device='cuda',
            wandb=opt.wandb,
            tensorboard=opt.tensorboard,
            class_names=class_names,
            notes=f"Model: {opt.model}, Stream: {opt.stream_mode}",
            grad_accum_steps=2,
            compute_val_loss=True,
            log_per_class_metrics=True,
            # lr=5e-5,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()
