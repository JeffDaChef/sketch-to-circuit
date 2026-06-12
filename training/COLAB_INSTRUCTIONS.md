# Training the Circuit Detector on Google Colab

## 1. What this step does — and why Colab?

**What it does:** Training teaches the YOLO11-nano neural network to recognise the
8 circuit component types (capacitor, diode, ground, junction, resistor, switch,
text, voltage\_source) in hand-drawn schematics. You feed it thousands of labelled
examples and it adjusts its internal numbers until it can label new images on its
own. The end result is a single file, `best.pt`, that contains everything the
network learned.

**Why Colab, not your Mac?** Training does enormous amounts of floating-point maths
— going through every image 100 times. An NVIDIA GPU (Graphics Processing Unit)
does this ~50× faster than a CPU. Google Colab gives you a free NVIDIA T4 GPU in
your browser with zero installation. Your Mac has an AMD GPU, but making PyTorch
talk to it on macOS via ROCm is unreliable and time-consuming — it is not worth the
hassle here. After training you just download `best.pt` to your Mac; running the
detector on single images works fine on CPU.

---

## 2. Before you start — prepare the dataset on your Mac

The training script needs a folder of images + labels in YOLO format. The
`cghd_prep.py` script creates that for you.

**Step 1 — Run the prep script in your terminal:**

```bash
python data_collection/cghd_prep.py --src <CGHD_folder> --out cghd_yolo
```

Replace `<CGHD_folder>` with the path to the folder where you downloaded the CGHD
dataset. For example:

```bash
python data_collection/cghd_prep.py --src ~/Downloads/CGHD --out cghd_yolo
```

This produces a folder called `cghd_yolo/` containing `data.yaml`, plus
`images/train`, `images/val`, `images/test` and matching `labels/` folders.

**Step 2 — Zip the folder** so it is a single file to upload:

```bash
zip -r cghd_yolo.zip cghd_yolo/
```

You should now have a file called `cghd_yolo.zip` in your working directory.
Check its size — it will probably be several hundred MB. If it is over ~1 GB, see
the Google Drive note in Step 3.

---

## 3. Training on Colab — step by step

### 3a. Open a new notebook

1. Go to **[colab.research.google.com](https://colab.research.google.com)** in
   your browser.
2. Click **New notebook** (or File → New notebook if you are already inside Colab).

### 3b. Switch to a free GPU

1. In the top menu, click **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, choose **T4 GPU**.
3. Click **Save**.

> The T4 GPU is what makes training fast. Without this step, training falls back
> to CPU and takes many hours instead of 1-2.

### 3c. Upload your dataset

**Option A — Direct upload (works well for files up to ~500 MB):**

Paste this into a new code cell and run it (press **Shift+Enter**):

```python
from google.colab import files
files.upload()   # a file picker will appear — select cghd_yolo.zip
```

**Option B — Google Drive mount (better for large files or if upload is slow):**

1. Upload `cghd_yolo.zip` to your Google Drive first (drag it into
   [drive.google.com](https://drive.google.com)).
2. In Colab, run:

```python
from google.drive import drive
drive.mount('/content/drive')
```

Then copy it to the Colab workspace:

```python
import shutil
shutil.copy('/content/drive/MyDrive/cghd_yolo.zip', '/content/cghd_yolo.zip')
```

### 3d. Unzip the dataset

Run in a new cell:

```python
!unzip -q cghd_yolo.zip
```

After this, you should have a folder at `/content/cghd_yolo/` with `data.yaml`
inside it.

### 3e. Install ultralytics

```python
!pip install ultralytics
```

This takes about 30 seconds. You only need to do it once per Colab session.

### 3f. Fix the path inside data.yaml

`cghd_prep.py` writes an absolute `path:` into `data.yaml` pointing to a folder
on **your Mac**. In Colab that path does not exist. Fix it with one command:

```python
!sed -i 's|^path:.*|path: /content/cghd_yolo|' /content/cghd_yolo/data.yaml
```

This replaces whatever `path:` line is in the file with the correct Colab path.
You can verify it worked by running `!head -5 /content/cghd_yolo/data.yaml`.

### 3g. Run training

**Easy way — YOLO CLI** (simplest, no Python file to upload):

```python
!yolo detect train model=yolo11n.pt data=/content/cghd_yolo/data.yaml epochs=100 imgsz=640 batch=16 patience=20 plots=True name=cghd_yolo11n
```

**Alternative — upload and run train_colab.py:**

1. Click the folder icon in the left sidebar of Colab.
2. Drag `training/train_colab.py` from your Mac into the `/content/` folder.
3. Run:

```python
!python train_colab.py --data /content/cghd_yolo/data.yaml
```

Both methods do exactly the same thing.

### 3h. Wait for training to finish

- Training takes **roughly 1–2 hours** on a T4 GPU for 100 epochs.
- You will see a progress bar and live metrics (loss, mAP) updating in the cell
  output.
- **Keep the Colab browser tab open.** Colab disconnects idle sessions
  (ones where you close the tab or stop interacting for a long time). If it
  disconnects, see the Troubleshooting section below.

### 3i. Download best.pt

When training finishes, the best weights are saved at:

```
/content/runs/detect/cghd_yolo11n/weights/best.pt
```

Download it to your Mac:

```python
from google.colab import files
files.download('/content/runs/detect/cghd_yolo11n/weights/best.pt')
```

Your browser will prompt you to save the file (it is usually 5–10 MB).

---

## 4. After downloading best.pt

1. **Create the weights folder** in your repo if it does not exist yet:

   ```bash
   mkdir -p training/weights
   ```

2. **Move `best.pt`** (from your Downloads folder) into it:

   ```bash
   mv ~/Downloads/best.pt training/weights/best.pt
   ```

3. **Do not commit `best.pt` to git.** Weight files are large binary files; git
   tracks diffs of text files and handles large binaries poorly. Instead, store
   `best.pt` locally or in a dedicated model registry. You can add it to
   `.gitignore`:

   ```bash
   echo "training/weights/*.pt" >> .gitignore
   ```

4. **Evaluate the model** on the held-out test split to see how well it performs
   on handwriting it has never seen:

   ```bash
   python training/evaluate.py --weights training/weights/best.pt --data cghd_yolo/data.yaml
   ```

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| **CUDA out of memory** | Lower the batch size: add `batch=8` to the training command (or `--batch 8` for train_colab.py). |
| **"No labels found in train set"** | The `path:` in `data.yaml` is wrong. Re-run the `sed` command in Step 3f and check it points to `/content/cghd_yolo`. |
| **Session disconnected / runtime crashed** | Re-run all the setup cells (unzip, pip install, sed fix), then restart training from the `!yolo detect train ...` cell. Colab saves checkpoints; you can resume from the last one by adding `resume=True` to the training command. |
| **Training seems to stop early (before 100 epochs)** | That is intentional — early stopping (patience=20) halted because the model stopped improving. It is fine; `best.pt` already holds the best checkpoint. |
| **files.download() does nothing / times out** | Try the Files panel instead: click the folder icon in the left sidebar, navigate to `runs/detect/cghd_yolo11n/weights/`, right-click `best.pt`, and choose **Download**. |
