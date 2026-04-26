import cv2
import numpy as np
from pathlib import Path
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent))
from core.slicer import slice_image_inference
from core.anomalib_engine import AnomalibEngine
from run_pure_pipeline import dynamic_pure_crop

def quick_eval(ckpt_path, src_dir, slice_size=640, overlap=0.2):
    engine = AnomalibEngine()
    engine.load_model(ckpt_path)
    
    src_dir = Path(src_dir)
    results = {'normal': [], 'abnormal': []}
    
    for label in ['normal', 'abnormal']:
        test_dir = src_dir / f"test/{label}"
        if not test_dir.exists(): continue
        
        files = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
        # Only take first 5 images for quick evaluation
        files = files[:5]
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
            logger.info(f"  [{label}] File: {f.name} -> Score: {score:.2f}")

if __name__ == '__main__':
    src = r"f:\Bear\apple\dataset\anomalib_P03_yiwu_sift"
    ckpt = r"f:\Bear\apple\runs\anomalib_pure\patchcore.ckpt"
    quick_eval(ckpt, src)
