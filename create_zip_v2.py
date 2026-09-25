from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

files = [
    "requirements.txt",
    "run_kaggle_profiler_v2.py",
    "src/__init__.py",
    "src/config.py",
    "src/preprocessing.py",
    "src/storage.py",
    "src/memory.py",
    "src/disk_blocking.py",
]

root = Path(".")
zip_path = root / "amazon_ml_kaggle_profiler_v2.zip"

with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
    for rel in files:
        p = root / rel
        assert p.exists(), f"Missing: {p}"
        # Force forward slashes for Kaggle Unix environment
        z.write(p, arcname=rel.replace("\\", "/"))

with ZipFile(zip_path, "r") as z:
    names = z.namelist()
    print("Files in ZIP:")
    print("\n".join(names))
    assert all("\\" not in n for n in names)
    assert set(names) == set(files)

print(f"\nVerified ZIP: {zip_path.absolute()}")
print(f"Size: {zip_path.stat().st_size / 1024:.2f} KB")
