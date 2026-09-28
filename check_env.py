"""Check the vpp environment is ready. Run: python check_env.py"""

import importlib
import sys

PACKAGES = [
    ("torch", True), ("torchvision", True), ("fastapi", True),
    ("PIL", True), ("numpy", True),
    # Cluster-only packages: missing on a laptop is fine
    ("openslide", False), ("timm", False), ("h5py", False),
    ("pandas", False), ("sklearn", False), ("cv2", False),
]

ok = True
print(f"Python {sys.version.split()[0]}")

for name, required in PACKAGES:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        print(f"  [ok]      {name} {version}")
    except Exception as error:
        tag = "MISSING" if required else "skip"
        print(f"  [{tag:7}] {name}: {error.__class__.__name__}")
        ok = ok and not required

try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s))")
        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum().item()
        print("GPU test: matrix multiply ran")
    else:
        print("GPU: none visible (expected on a laptop; on the cluster, "
              "run this inside a GPU job)")
except Exception as error:
    ok = False
    print(f"GPU test failed: {error}")

try:
    import openslide
    print(f"OpenSlide library: {openslide.__library_version__}")
except Exception:
    pass

print("\nEnvironment ready." if ok else "\nSomething required is missing; see above.")
sys.exit(0 if ok else 1)
