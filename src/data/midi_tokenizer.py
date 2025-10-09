import pretty_midi
import numpy as np

STEPS_PER_BEAT = 24
MAX_TIME_SHIFT_STEPS = 100
RECONSTRUCT_BPM = 120.0

class MIDITokenizer:
    def __init__(self):
        self.event_to_int: dict[str, int] = {}
        self.int_to_event: dict[int, str] = {}
        self._build_vocabulary()

    def _build_vocabulary(self):
        vocab_idx = 0
        # Note-On events (128 pitches)
        for pitch in range(128):
            self.event_to_int[f"Note-On_{pitch}"] = vocab_idx
            self.int_to_event[vocab_idx] = f"Note-On_{pitch}"
            vocab_idx += 1
        # Note-Off events (128 pitches)
        for pitch in range(128):
            self.event_to_int[f"Note-Off_{pitch}"] = vocab_idx
            self.int_to_event[vocab_idx] = f"Note-Off_{pitch}"
            vocab_idx += 1
        # Velocity events (binned into 32 levels)
        for velocity_bin in range(32):
            self.event_to_int[f"Velocity_{velocity_bin}"] = vocab_idx
            self.int_to_event[vocab_idx] = f"Velocity_{velocity_bin}"
            vocab_idx += 1
        # Time-Shift events
        for time_step in range(1, MAX_TIME_SHIFT_STEPS + 1):
            self.event_to_int[f"Time-Shift_{time_step}"] = vocab_idx
            self.int_to_event[vocab_idx] = f"Time-Shift_{time_step}"
            vocab_idx += 1

    @staticmethod
    def _velocity_to_bin(velocity: int) -> int:
        return int(np.clip(velocity, 1, 127) * 32 // 128)

    @staticmethod
    def _bin_to_velocity_center(bin_idx: int) -> int:
        return int(np.clip(bin_idx * 4 + 2, 1, 127))

    @staticmethod
    def _seconds_per_step(bpm: float = RECONSTRUCT_BPM) -> float:
        return (60.0 / bpm) / STEPS_PER_BEAT

    @staticmethod
    def _quantize_time_to_step(pm: pretty_midi.PrettyMIDI, t: float) -> int:
        try:
            tempo = pm.get_tempo_changes()[1][0]
        except IndexError:
            tempo = 120.0
        seconds_per_beat = 60.0 / tempo
        seconds_per_step = seconds_per_beat / STEPS_PER_BEAT
        return 0 if seconds_per_step == 0 else int(round(t / seconds_per_step))

    def midi_to_tokens(self, midi_file_path: str) -> list[int]:
        pm = pretty_midi.PrettyMIDI(midi_file_path)
        events: list[tuple[int, int, int, int]] = []
        for inst in pm.instruments:
            for note in inst.notes:
                start_step = self._quantize_time_to_step(pm, note.start)
                end_step = self._quantize_time_to_step(pm, note.end)
                if end_step <= start_step:
                    end_step = start_step + 1
                v_bin = self._velocity_to_bin(note.velocity)
                events.append((start_step, 1, note.pitch, v_bin))
                events.append((end_step, 0, note.pitch, 0))
        if not events:
            return []
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        tokens_str: list[str] = []
        current_step = 0
        i = 0
        while i < len(events):
            step, _, _, _ = events[i]
            dt = step - current_step
            while dt > 0:
                chunk = min(dt, MAX_TIME_SHIFT_STEPS)
                tokens_str.append(f"Time-Shift_{chunk}")
                dt -= chunk
                current_step += chunk
            same_time_events = []
            while i < len(events) and events[i][0] == step:
                same_time_events.append(events[i])
                i += 1
            for (_, k, p, v) in same_time_events:
                if k == 0:
                    tokens_str.append(f"Note-Off_{p}")
            for (_, k, p, v) in same_time_events:
                if k == 1:
                    tokens_str.append(f"Velocity_{v}")
                    tokens_str.append(f"Note-On_{p}")
        try:
            return [self.event_to_int[tok] for tok in tokens_str]
        except KeyError as e:
            raise ValueError(f"Encountered unknown token while encoding: {e}")

    def tokens_to_midi(self, tokens: list[int]) -> pretty_midi.PrettyMIDI:
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, is_drum=False)
        current_step = 0
        current_velocity_bin = 16
        secs_per_step = self._seconds_per_step(RECONSTRUCT_BPM)
        active: dict[int, list[tuple[int, int]]] = {p: [] for p in range(128)}
        for tid in tokens:
            if tid not in self.int_to_event:
                continue
            ev = self.int_to_event[tid]
            kind, val_str = ev.split("_", 1)
            val = int(val_str)
            if kind == "Time-Shift":
                if val > 0:
                    current_step += val
            elif kind == "Velocity":
                current_velocity_bin = int(np.clip(val, 0, 31))
            elif kind == "Note-On":
                pitch = int(np.clip(val, 0, 127))
                active[pitch].append((current_step, current_velocity_bin))
            elif kind == "Note-Off":
                pitch = int(np.clip(val, 0, 127))
                if active[pitch]:
                    start_step, vbin = active[pitch].pop(0)
                    end_step = max(current_step, start_step + 1)
                    note = pretty_midi.Note(
                        velocity=self._bin_to_velocity_center(vbin),
                        pitch=pitch,
                        start=start_step * secs_per_step,
                        end=end_step * secs_per_step,
                    )
                    instrument.notes.append(note)
        last_step = current_step + 1
        for pitch in range(128):
            while active[pitch]:
                start_step, vbin = active[pitch].pop(0)
                note = pretty_midi.Note(
                    velocity=self._bin_to_velocity_center(vbin),
                    pitch=pitch,
                    start=start_step * secs_per_step,
                    end=max(
                        start_step * secs_per_step + secs_per_step,
                        last_step * secs_per_step
                    ),
                )
                instrument.notes.append(note)
        instrument.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments.append(instrument)
        return pm
    
    
if __name__ == '__main__':
    tokenizer = MIDITokenizer()
    print("\nSample of event-to-integer mapping: ")
    for i in range(5):
        event = tokenizer.int_to_event[i]
        integer = tokenizer.event_to_int[event]
        print(f"'{event}' -> {integer}")