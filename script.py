import torch
import torch.onnx
import onnxruntime as ort
import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.transcribe import transcribe
from whisper.decoding import decode
# Create a dummy input for ONNX export
# Whisper expects audio as a Mel spectrogram tensor
# Here we use a random tensor with the expected shape: (batch_size, n_mels, n_frames)
dummy_audio_features_input = torch.randn(1, 1500, 384)
dummy_tokens_input = torch.randn(1, 3)

# Load the Whisper model
# Options: "tiny", "base", "small", "medium", "large"
# model = whisper.load_model("tiny")
num_languages = 99

# class DummyModel:
#     def __init__(self, session: ort.InferenceSession):
#         self.session = session
#         self.is_multilingual = True
#         self.num_languages = 99

#     def set_alignment_heads(self, _dump: bytes) -> None:  # no-op for ONNX models
#         return

#     def transcribe(self, *args, **kwargs):
#         return transcribe_function(self, *args, **kwargs)

#     def decode(self, *args, **kwargs):
#         return decode_function(self, *args, **kwargs)

#     def detect_language(self, *args, **kwargs):
#         return detect_language_function(self, *args, **kwargs)


dims = ModelDimensions(n_mels=80,
                       n_audio_ctx=1500,
                       n_audio_state=384,
                       n_audio_head=6,
                       n_audio_layer=4,
                       n_vocab=51865,
                       n_text_ctx=448,
                       n_text_state=384,
                       n_text_head=6,
                       n_text_layer=4)

# dummy_model = DummyModel()
# dummy_model.decode = model.decode
# dummy_model.dims = dims
# dummy_model.is_multilingual = True
# dummy_model.num_languages = model.num_languages
# dummy_model.forward = lambda x: x

# # Transcribe the audio file
# result = transcribe(dummy_model, "data/audio.ogg", language="es")

# # Print the transcription
# print("Transcription:")
# print(result["text"])

# Export to ONNX
onnx_path = "model.onnx"
# onnx_program = torch.onnx.export(
#     dummy_model,                      # PyTorch model
#     dummy_audio_features_input,
#     dynamo=True
# )

# onnx_program.optimize()
# onnx_program.save(onnx_path)

# torch.onnx.export(
#     dummy_model,                      # PyTorch model
#     dummy_audio_features_input,
#     f=onnx_path,                  # Output path
#     export_params=True,         # Store trained parameters
#     opset_version=11,           # ONNX version
#     do_constant_folding=True,   # Optimize constant folding
#     input_names=['input'],      # Input names
#     output_names=['output'],    # Output names
#     dynamic_axes={
#         'input': {0: 'batch_size'},    # Dynamic batch size
#         'output': {0: 'batch_size'}
#     }
# )

# print(f"Model exported to {onnx_path}\n")

# Verify the ONNX model
ort_session = ort.InferenceSession(onnx_path)
# model = Whisper(ort_session, dims)
# model.dims = dims
ort_session.dims = dims
ort_session.is_multilingual = True
ort_session.num_languages = num_languages
ort_session.decode = decode

# Test with the same function
# Transcribe the audio file
result = transcribe(ort_session, "data/audio.ogg", language="es")

# Print the transcription
print("Transcription:")
print(result["text"])

print("ONNX model verification successful!")
