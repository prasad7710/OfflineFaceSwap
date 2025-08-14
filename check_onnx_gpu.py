import onnxruntime as ort

# Print version if available
if hasattr(ort, '__version__'):
    print("✅ ONNX Runtime version:", ort.__version__)
else:
    print("ℹ️  ONNX version info not available.")

try:
    # Load a dummy model (you can download a real one if needed)
    sess = ort.InferenceSession("inswapper_128.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("✅ GPU is working! Using:", sess.get_providers())
except Exception as e:
    print("❌ Could not use GPU. Error:\n", str(e))
