import sounddevice as sd

default_input = sd.default.device[0]
print(f"Default input device index: {default_input}")
print("Device info:")
print(sd.query_devices(default_input))
