import sys
import importlib.util

packages = [
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "accelerate",
    "numpy",
]

print("=== Python ===")
print(sys.version)
print("executable:", sys.executable)

print("\n=== Packages ===")
for pkg in packages:
    spec = importlib.util.find_spec(pkg)
    if spec is None:
        print(f"BLOCK: {pkg} not installed")
        raise SystemExit(1)

    module = __import__(pkg)
    version = getattr(module, "__version__", "unknown")
    path = getattr(module, "__file__", "unknown")
    print(f"{pkg}: {version}")
    print(f"  file: {path}")

print("\nPASS: environment import check OK")
