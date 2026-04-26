import cv2
import numpy as np
from pathlib import Path
import shutil
from loguru import logger
import sys

# Ensure imports work if run from project root
sys.path.append(str(Path(__file__).parent))

from core.slicer import slice_image_inference
from core.anomalib_engine import AnomalibEngine

def dynamic_pure_crop(image: np.ndarray, shrink_ratio: float = 0.05) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        margin_x = int(w * shrink_ratio)
        margin_y = int(h * shrink_ratio)
        return image[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]
    return image

def step2_prepare_data(src_dir, dst_dir, slice_size=640, overlap=0.2):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
        
    for split in ['train/normal', 'test/normal', 'test/abnormal']:
        src_split = src_dir / split
        dst_split = dst_dir / split
        dst_split.mkdir(parents=True, exist_ok=True)
        
        if not src_split.exists(): continue
        
        files = list(src_split.glob('*.png')) + list(src_split.glob('*.jpg'))
        if split == 'train/normal' and len(files) > 40:
            import random
            random.seed(42)  # For reproducibility
            files = random.sample(files, 40)
            
        logger.info(f"Processing {split}: {len(files)} files")
        
        total_slices = 0
        for idx, f in enumerate(files):
            # Load with cv2.imdecode for chinese paths
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            cropped = dynamic_pure_crop(img)
            img_h, img_w = cropped.shape[:2]
            
            # slice
            if min(img_w, img_h) > 2 * slice_size:
                slices = slice_image_inference(cropped, slice_size=slice_size, overlap=overlap)
                for s_idx, (slice_img, _) in enumerate(slices):
                    out_name = dst_split / f"{f.stem}_slice{s_idx}{f.suffix}"
                    is_success, buffer = cv2.imencode(out_name.suffix, slice_img)
                    if is_success:
                        buffer.tofile(str(out_name))
                total_slices += len(slices)
            else:
                out_name = dst_split / f.name
                is_success, buffer = cv2.imencode(out_name.suffix, cropped)
                if is_success:
                    buffer.tofile(str(out_name))
                total_slices += 1
                
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx+1}/{len(files)} images...")
                
        logger.info(f"Finished {split}. Total slices generated: {total_slices}")

def step3_train_baseline(data_dir):
    engine = AnomalibEngine()
    # Ensure to turn off geometric augmentations!
    ckpt = engine.train(
        data_dir=str(data_dir),
        output_dir="runs/anomalib_pure",
        backbone="wide_resnet50_2",
        image_size=256,
        train_batch_size=16,
        eval_batch_size=16,
        aug_enabled=False,
        max_epochs=1
    )
    return ckpt

def step4_evaluate(ckpt_path, src_dir, slice_size=640, overlap=0.2):
    engine = AnomalibEngine()
    engine.load_model(ckpt_path)
    
    src_dir = Path(src_dir)
    results = {'normal': [], 'abnormal': []}
    
    for label in ['normal', 'abnormal']:
        test_dir = src_dir / f"test/{label}"
        if not test_dir.exists(): continue
        
        files = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
        logger.info(f"Evaluating {label}: {len(files)} images")
        
        for idx, f in enumerate(files):
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            
            cropped = dynamic_pure_crop(img)
            img_h, img_w = cropped.shape[:2]
            
            if min(img_w, img_h) > 2 * slice_size:
                slices = slice_image_inference(cropped, slice_size, overlap)
                tile_imgs = [s[0] for s in slices]
                tile_coords = [(c.x_start, c.y_start, c.x_start + c.width, c.y_start + c.height) for _, c in slices]
                
                tile_results = engine.predict_tiles(tile_imgs, (img_h, img_w), tile_coords)
                full_map = engine.assemble_anomaly_map(tile_results, (img_h, img_w), slice_size, overlap)
                score = float(full_map.max())
            else:
                _, score = engine.predict_single(cropped)
                
            results[label].append(score)
            if (idx + 1) % 10 == 0:
                logger.info(f"  Evaluated {idx+1}/{len(files)} images...")
            
    norm_scores = results['normal']
    abnorm_scores = results['abnormal']
    
    if not norm_scores or not abnorm_scores:
        logger.warning("Missing normal or abnormal scores")
        return
        
    mean_norm = np.mean(norm_scores)
    mean_abnorm = np.mean(abnorm_scores)
    
    # Simple F1 Calculation: threshold search
    best_f1 = 0
    best_thresh = 0
    all_scores = norm_scores + abnorm_scores
    for thresh in np.linspace(min(all_scores), max(all_scores), 100):
        tp = sum(1 for s in abnorm_scores if s >= thresh)
        fp = sum(1 for s in norm_scores if s >= thresh)
        fn = sum(1 for s in abnorm_scores if s < thresh)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    # Calculate separation: D = (mean_abnorm - mean_norm) / (std_norm + std_abnorm)
    std_norm = np.std(norm_scores)
    std_abnorm = np.std(abnorm_scores)
    separation = (mean_abnorm - mean_norm) / (std_norm + std_abnorm + 1e-6)
            
    logger.info("=== Evaluation Results ===")
    logger.info(f"Normal Mean Score  : {mean_norm:.2f}")
    logger.info(f"Defect Mean Score  : {mean_abnorm:.2f}")
    logger.info(f"Separation Index   : {separation:.2f}")
    logger.info(f"Best F1-Score      : {best_f1:.3f} (at threshold {best_thresh:.2f})")

if __name__ == '__main__':
    src = r"f:\Bear\apple\dataset\anomalib_P03_yiwu_sift"
    dst = r"f:\Bear\apple\dataset\anomalib_P03_yiwu_pure_sliced"
    
    logger.info("Starting Step 2: Preparing Pure Sliced Data...")
    step2_prepare_data(src, dst)
    
    logger.info("Starting Step 3: Baseline Training...")
    ckpt = step3_train_baseline(dst)
    
    if ckpt:
        logger.info("Starting Step 4: Evaluation...")
        step4_evaluate(ckpt, src)
    else:
        logger.error("Training failed, no checkpoint generated.")
