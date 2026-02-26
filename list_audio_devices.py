import sounddevice as sd

print("=== Input Devices ===")
for idx, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f"{idx}: {dev['name']} (inputs: {dev['max_input_channels']})")

print("\nSet the 'audio_device' variable in config/settings.py to the index or name of your preferred device above.")
print("Example: audio_device = 1  # or 'USB Audio'")
