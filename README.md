# Data266 Gen AI Fall Detection

Pipeline for downloading UR Fall Detection data, validating fall files, extracting archives, and generating pose features.

## Project Structure

- `data.py` downloads ADL and FALL `cam0-rgb` sequences and extracts them.
- `extract_zips.py` extracts any remaining zip files in `data/sample_videos`.
- `repair_falls_data.py` repairs missing/corrupt files for `data/falls/fall-01..fall-08`.
- `verify_falls_data.py` validates CSV/ZIP integrity in `data/falls`.
- `pose_features.py` runs MediaPipe pose landmark extraction and writes JSON features.

## Setup

```bash
cd /Users/#/Desktop/Gen_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Order (Recommended)

### 1) Download dataset files

```bash
python data.py
```

### 2) Repair fall files (if any are missing/corrupt)

```bash
python repair_falls_data.py
```

### 3) Verify fall data integrity

```bash
python verify_falls_data.py
```

### 4) Extract any leftover zips from sample folder

```bash
python extract_zips.py
```

### 5) Generate pose features

```bash
python pose_features.py
```

Output JSON files are saved in:

- `data/pose_features/`






## Notes

- `data/` is ignored by git in `.gitignore`.
- If you see `invalid signature (likely HTML or partial download)`, run `repair_falls_data.py` and then `verify_falls_data.py` again.
