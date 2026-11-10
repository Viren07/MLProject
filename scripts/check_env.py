"""Sanity check: run this after setup. `python scripts/check_env.py`"""
import importlib
import os
import sys

REQUIRED = ["numpy", "scipy", "pandas", "PIL", "matplotlib", "tqdm", "yaml",
            "dotenv", "imagecorruptions", "openai"]


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    ok = sys.version_info >= (3, 10)
    if not ok:
        print("FAIL: need Python 3.10 or newer")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"{'ok   ' if in_venv else 'WARN '} virtual env active: {in_venv} ({sys.prefix})")
    if not in_venv:
        print("      Packages would come from your global Python. Activate .venv first.")
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL  {name}: {e}")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    print("ok    OPENAI_API_KEY set" if os.getenv("OPENAI_API_KEY")
          else "WARN  OPENAI_API_KEY not set (copy .env.example to .env)")

    try:
        import numpy as np
        from imagecorruptions import corrupt
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = corrupt(img, corruption_name="defocus_blur", severity=3)
        print(f"ok    corruption smoke test, output shape {out.shape}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL  corruption smoke test: {e}")

    print("\nEnvironment looks good." if ok else "\nFix the FAIL lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
