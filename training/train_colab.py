"""YOLO11-nano training script (meant to run on Google Colab).

WHY THIS EXISTS
---------------
"Training" is the process of showing the neural network thousands of labelled
examples — images of hand-drawn circuits where every component (resistor,
capacitor, etc.) has a box drawn around it and a name attached — and letting the
network slowly adjust its internal numbers until it can draw those boxes and names
itself on images it has never seen.

The result of training is a single file called `best.pt`. That file IS the
trained detector: all the network's learned knowledge is packed into it. Once you
have `best.pt` you can copy it anywhere and run it on new images without touching
the training code again.

WHY COLAB, NOT YOUR MAC
-----------------------
Training goes through every image hundreds of times (that is what "epochs" means),
doing heavy floating-point math each time. A dedicated GPU (Graphics Processing
Unit) does that math ~50× faster than a CPU. Your Mac has an AMD GPU but getting
PyTorch to talk to it via ROCm is finicky and unreliable on macOS. Google Colab
gives you a free NVIDIA T4 GPU in a browser with zero setup — it is the right
tool for this step. After training you just download `best.pt` and run inference
on your Mac's CPU, which is plenty fast for single images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Guard the ultralytics import so a missing install gives a friendly message
# instead of a confusing ModuleNotFoundError stack trace.
try:
    from ultralytics import YOLO
except ImportError:
    print(
        "\n[ERROR] The 'ultralytics' package is not installed.\n"
        "Fix it by running:  pip install ultralytics\n"
        "If you are in Colab, run:  !pip install ultralytics\n"
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11-nano detector on the CGHD circuit dataset."
    )

    # --data is the one thing we cannot guess: it is the path to the data.yaml
    # file produced by cghd_prep.py that tells YOLO where the images and labels live.
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data.yaml file produced by cghd_prep.py.",
    )

    # YOLO11-nano ('yolo11n.pt') is the lightest model in the YOLO11 family.
    # It trains fast and is accurate enough for the 8-class circuit dataset.
    # On first run ultralytics will auto-download the file (~2 MB).
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Starting checkpoint. Default is yolo11n.pt (YOLO11-nano pretrained on COCO).",
    )

    # One epoch = one full pass through the training images. 100 is a good
    # default for a small dataset; early stopping (--patience) will quit sooner
    # if the model stops improving.
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs (default 100).",
    )

    # Image size: all images are resized to --imgsz × --imgsz squares before
    # being fed to the network. 640 is the standard YOLO size; smaller (e.g. 320)
    # trains faster but is less accurate on small components.
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (square). Default 640.",
    )

    # Batch size = how many images the GPU processes at once. 16 fits easily on
    # a Colab T4. If you see an "out of memory" error, lower this to 8.
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Number of images per training batch. Lower to 8 if you run out of GPU memory.",
    )

    # Where YOLO writes its output folders. The full path will be
    #   <project>/<name>/weights/best.pt
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Parent folder for all training outputs. Default 'runs/detect' "
             "(matches the bare `yolo detect train` layout and COLAB_INSTRUCTIONS.md).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="cghd_yolo11n",
        help="Sub-folder name for this specific run. Default 'cghd_yolo11n'.",
    )

    # Patience = how many epochs in a row with no improvement before training
    # stops early. This saves time when the model has already converged.
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience: stop if no improvement for this many epochs. Default 20.",
    )

    args = parser.parse_args()

    # Resolve the data.yaml path so any relative path works from any working dir.
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"\n[ERROR] data.yaml not found at: {data_path}")
        print("Did you run cghd_prep.py first?  python data_collection/cghd_prep.py --src <CGHD_folder> --out cghd_yolo")
        sys.exit(1)

    # Load the starting weights. 'yolo11n.pt' was pretrained on COCO, so the
    # network already knows basic visual features (edges, textures, shapes).
    # Fine-tuning from this checkpoint is MUCH faster than training from scratch.
    print(f"\n[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Starting training — this will take ~1-2 hours on a Colab T4 GPU.")
    print(f"[INFO] Output will appear in:  {args.project}/{args.name}/\n")

    # Call ultralytics' built-in training loop. It handles data loading,
    # augmentation, loss computation, validation, and checkpointing automatically.
    # plots=True saves accuracy/loss curves as PNG files in the run folder,
    # which are handy for checking whether training went well.
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        plots=True,
        seed=0,            # reproducible training run (the drafter split is seeded too)
    )

    # Tell the user exactly where to find the trained weights file.
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best weights saved to:  {best_pt}")
    print()
    print("NEXT STEP: Download that file to your Mac and put it at:")
    print("  training/weights/best.pt")
    print()
    print("In Colab, run this cell to download it:")
    print("  from google.colab import files")
    print(f"  files.download('{best_pt}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
