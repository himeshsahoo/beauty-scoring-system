"""Create a leak-aware crop manifest from an official SCUT-FBP5500 download."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2, pandas as pd
from common import FaceDetector, landmark_box, load_config, read_landmarks, resolve, select_largest

def find_images(root: Path):
    return {p.name: p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

def find_labels(root: Path, configured: Path) -> Path:
    if configured.exists(): return configured
    candidates = list(root.rglob("*.csv")) + list(root.rglob("*.txt"))
    if not candidates: raise FileNotFoundError("No ratings CSV/TXT found. See README dataset setup.")
    return candidates[0]

def read_labels(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2: raw = pd.read_csv(path, sep=r"\s+", header=None)
    raw = raw.iloc[:, :2]; raw.columns = ["image_id", "score_1_5"]
    raw["image_id"] = raw.image_id.astype(str).str.strip(); raw["score_1_5"] = pd.to_numeric(raw.score_1_5, errors="coerce")
    raw = raw.dropna(); raw = raw[raw.score_1_5.between(1,5)]
    if raw.empty: raise ValueError("Could not parse 1–5 SCUT ratings from label file.")
    raw["score_1_10"] = 1 + (raw.score_1_5 - 1) * 9 / 4
    # SCUT images represent one portrait per subject; preserving any provided identity/group is supported downstream.
    raw["identity"] = raw.image_id.str.replace(r"\.[^.]+$", "", regex=True)
    return raw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--overwrite", action="store_true"); args=ap.parse_args()
    cfg=load_config(args.config); root=resolve(cfg,cfg["data"]["root"]); out=resolve(cfg,cfg["data"]["processed_manifest"]); crops=resolve(cfg,cfg["data"]["crops_dir"])
    if not root.exists(): raise FileNotFoundError(f"Dataset missing at {root}. Run download_dataset.py first.")
    if out.exists() and not args.overwrite: print(f"Manifest already exists: {out}; use --overwrite to recreate."); return
    labels=read_labels(find_labels(root,resolve(cfg,cfg["data"]["labels_file"]))); images=find_images(root); detector=FaceDetector(cfg); crops.mkdir(parents=True,exist_ok=True); rows=[]
    for _, r in labels.iterrows():
        image_path=images.get(r.image_id)
        if not image_path: continue
        bgr=cv2.imread(str(image_path));
        if bgr is None: continue
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); boxes=detector.detect(rgb); landmark_path=root / "facial landmark" / f"{Path(r.image_id).stem}.pts"; landmarks=read_landmarks(landmark_path)
        if boxes:
            box=select_largest(boxes)
        elif landmarks is not None:
            box=landmark_box(landmarks,rgb.shape,detector.margin)
        else:
            continue
        try:
            crop=detector.crop(rgb,box,landmarks=landmarks)
        except Exception:
            continue
        dest=crops / r.image_id
        cv2.imwrite(str(dest),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)); rows.append({**r.to_dict(),"source_path":str(image_path),"landmark_path":str(landmark_path) if landmark_path.exists() else None,"crop_path":str(dest),"detected_faces":len(boxes),"face_box":str(box)})
    frame=pd.DataFrame(rows); out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False)
    print(f"Prepared {len(frame)}/{len(labels)} usable face crops. Manifest: {out}")
if __name__ == "__main__": main()
