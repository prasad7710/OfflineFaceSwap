import ctypes
try:
    ctypes.WinDLL("cublasLt64_12.dll")
    print("✅ cublasLt64_12.dll loaded successfully!")
except OSError as e:
    print("❌ Failed to load cublasLt64_12.dll:", e)
