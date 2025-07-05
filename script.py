import torch
import torch.onnx
import onnxruntime as ort
import whisper
from whisper.transcribe import transcribe
# Create a dummy input for ONNX export
# Whisper expects audio as a Mel spectrogram tensor
# Here we use a random tensor with the expected shape: (batch_size, n_mels, n_frames)
dummy_input = torch.randn(1, 80, 3000)

# Load the Whisper model
# Options: "tiny", "base", "small", "medium", "large"
model = whisper.load_model("tiny")
model.forward

# Transcribe the audio file
result = transcribe(model, "data/audio.ogg")

# Print the transcription
print("Transcription:")
print(result["text"])

# Export to ONNX
onnx_path = "model.onnx"

torch.onnx.export(
    model,                      # PyTorch model
    f=onnx_path,                  # Output path
    export_params=True,         # Store trained parameters
    opset_version=11,           # ONNX version
    do_constant_folding=True,   # Optimize constant folding
    input_names=['input'],      # Input names
    output_names=['output'],    # Output names
    dynamic_axes={
        'input': {0: 'batch_size'},    # Dynamic batch size
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to {onnx_path}")