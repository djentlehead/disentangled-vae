import pretty_midi
import inspect

try:
    print(f"pretty_midi version: {pretty_midi.__version__}")
    print(f"File location: {pretty_midi.__file__}")
    print("\nSignature for pretty_midi.PrettyMIDI.synthesize:")
    print(inspect.signature(pretty_midi.PrettyMIDI.synthesize))

except Exception as e:
    print(f"An error occurred: {e}")