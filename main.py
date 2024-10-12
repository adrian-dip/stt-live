import pyaudio
import numpy as np

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
VOLUME = 1.1

def apply_tremolo(audio_data, rate, depth):

    t = np.arange(len(audio_data)) / 44100  # Assume 44100 Hz sample rate
    mod = 1 + depth * np.sin(2 * np.pi * rate * t)
    
    return (audio_data * mod).astype(np.int16)

def apply_distortion(audio_data, amount):

    normalized = audio_data.astype(np.float32) / 32767.0
    
    distorted = np.tanh(normalized * amount * 10) / np.tanh(amount * 10)
    
    return (distorted * 32767).astype(np.int16)

def apply_echo(audio_data, delay, decay):

    delayed = np.zeros_like(audio_data)
    delayed[delay:] = audio_data[:-delay]
    
    echo = audio_data + (delayed * decay).astype(np.int16)
    
    return echo

def apply_pitch_shift(audio_data, semitones):

    factor = 2 ** (semitones / 12)
    
    t = np.arange(len(audio_data)) / factor
    
    shifted = np.interp(t, np.arange(len(audio_data)), audio_data)
    
    return shifted.astype(np.int16)

def apply_bitcrusher(audio_data, bits):

    bits = max(1, min(16, bits))
    scale = (2 ** bits) / 65536
    crushed = (audio_data * scale).astype(np.int16) * (1 / scale)
    
    return crushed.astype(np.int16)

p = pyaudio.PyAudio()

input_stream = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)

output_stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True,
                       frames_per_buffer=CHUNK)

print("* Recording and playing back. Press Ctrl+C to stop.")

try:
    while True:
        data = input_stream.read(CHUNK)

        audio_data = np.frombuffer(data, dtype=np.int16)

        #audio_data = (audio_data * VOLUME).astype(np.int16)

        #audio_data = apply_bitcrusher(audio_data, bits=4)
        
        data = audio_data.tobytes()
        
        output_stream.write(data)

except KeyboardInterrupt:
    print("* Stopped recording")

finally:
    # Stop and close both streams
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    
    # Terminate PyAudio
    p.terminate()