import cv2
import numpy as np
from pathlib import Path

def analyze():
    base_dir = Path(r"f:\Bear\apple\dataset\anomalib_P03_yiwu_sift")
    if not base_dir.exists():
        print(f"Error: {base_dir} not found")
        return

    files = list(base_dir.rglob("*.jpg")) + list(base_dir.rglob("*.png"))
    print(f"Found {len(files)} files.")
    
    if not files:
        return

    max_t, max_b, max_l, max_r = 0, 0, 0, 0
    img_h, img_w = 0, 0

    count = 0
    for p in files:
        # Load image via numpy to handle Chinese paths
        with open(p, "rb") as f:
            chunk = f.read()
        img = cv2.imdecode(np.frombuffer(chunk, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if img is None: continue
        
        if img_h == 0:
            img_h, img_w = img.shape
            print(f"Resolution: {img_w}x{img_h}")

        _, t = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        c = cv2.findNonZero(t)
        if c is not None:
            x, y, w, h = cv2.boundingRect(c)
            max_t = max(max_t, y)
            max_b = max(max_b, img_h - (y + h))
            max_l = max(max_l, x)
            max_r = max(max_r, img_w - (x + w))
        
        count += 1
        if count >= 100: # limit to 100 images for speed
            break

    print(f"Max borders across {count} images:")
    print(f"Top: {max_t}, Bottom: {max_b}, Left: {max_l}, Right: {max_r}")
    
    safe_t = int(max_t * 1.1) + 5
    safe_b = int(max_b * 1.1) + 5
    safe_l = int(max_l * 1.1) + 5
    safe_r = int(max_r * 1.1) + 5
    print(f"Recommended Safe ROI Margins: T={safe_t}, B={safe_b}, L={safe_l}, R={safe_r}")
    print(f"Final Crop Size: {img_w - safe_l - safe_r}x{img_h - safe_t - safe_b}")

if __name__ == "__main__":
    analyze()
