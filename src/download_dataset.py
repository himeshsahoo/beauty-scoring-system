"""Downloads the official non-commercial SCUT-FBP5500 archive via Google Drive."""
import argparse
from pathlib import Path
import gdown
from common import load_config, resolve
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--config",default=None); args=ap.parse_args(); cfg=load_config(args.config); root=resolve(cfg,cfg["data"]["root"]); root.parent.mkdir(parents=True,exist_ok=True)
 archive=root.parent/"SCUT-FBP5500_v2.1.zip"; gdown.download(id="1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf",output=str(archive),quiet=False); import zipfile
 with zipfile.ZipFile(archive) as z: z.extractall(root.parent)
 print(f"Downloaded and extracted under {root.parent}. Non-commercial research use only.")
if __name__=="__main__": main()
