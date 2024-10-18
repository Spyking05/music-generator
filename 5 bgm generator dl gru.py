import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from music21 import converter, note, chord, stream, instrument
import random
import pygame

# Load MIDI files from the directory and extract notes
def load_midi_files(midi_folder):
    notes = []
    midi_files = [os.path.join(midi_folder, f) for f in os.listdir(midi_folder) if f.endswith('.midi')]
    
    print(f"Looking for MIDI files in: {midi_folder}")
    print(f"Found {len(midi_files)} MIDI files.")
    
    if not midi_files:
        print("No MIDI files found in the folder.")
        return notes
    
    for file in midi_files:
        print(f"Parsing file: {file}")
        try:
            midi = converter.parse(file)
            for element in midi.flatten().notes:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.pitches))
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            continue  # Skip to the next file if there's an error
    
    print(f"Total notes/chords loaded: {len(notes)}")
    return notes

# Prepare sequences from the notes
def prepare_sequences(notes, sequence_length=100):
    if len(notes) < sequence_length:
        print("Not enough notes to create sequences.")
        return [], [], [], {}, {}

    unique_notes = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
    int_to_note = dict((number, note) for number, note in enumerate(unique_notes))
    
    sequences = []
    next_notes = []
    for i in range(len(notes) - sequence_length):
        seq = notes[i:i + sequence_length]
        sequences.append([note_to_int[note] for note in seq])
        next_notes.append(note_to_int[notes[i + sequence_length]])  # Next note
    
    print(f"Generated {len(sequences)} sequences.")
    return np.array(sequences), np.array(next_notes), unique_notes, note_to_int, int_to_note

# Build CNN-GRU model
def build_cnn_gru_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        GRU(512, return_sequences=True),
        Dropout(0.3),
        GRU(512),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    
    return model

# Train the CNN-GRU model
def train_cnn_gru_model(model, sequences, next_notes, epochs=0):
    sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))  # Reshape for CNN-GRU
    sequences = sequences / float(len(unique_notes))  # Normalize
    model.fit(sequences, next_notes, epochs=epochs, batch_size=64)

# Add drum and bass generation functions
def generate_drum_pattern(length=500):
    """Simple drum pattern: alternating kick and snare."""
    pattern = []
    for i in range(length):
        if i % 4 == 0:  # Kick on every 1st beat
            pattern.append("Kick")
        elif i % 4 == 2:  # Snare on every 3rd beat
            pattern.append("Snare")
        else:
            pattern.append("HiHat")  # Hi-hat on 2nd and 4th
    return pattern

def generate_bass_line(melody, scale, length=500):
    """Generate a bassline following the melody."""
    bass = []
    for i in range(length):
        melody_note = melody[i]
        if melody_note in scale:
            bass_note = scale[max(0, scale.index(melody_note) - 3)]  # Use a note below the melody in the scale
        else:
            bass_note = random.choice(scale)
        bass.append(bass_note)
    return bass

def generate_structured_music_with_drums_bass(model, sequences, sequence_length=100, length=500, diversity=1.0):
    """Generate structured music along with drums and bass."""
    major_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    generated_melody = []
    
    sequence = np.reshape(sequences[random.randint(0, len(sequences) - sequence_length - 1)], (1, sequence_length, 1)) / float(len(unique_notes))
    
    for _ in range(length):
        prediction = model.predict(sequence, verbose=0)
        
        # Extract the first batch prediction and flatten it to 1D
        prediction = prediction.flatten()
        
        # Adjust for diversity and normalize
        prediction = np.log(prediction + 1e-8) / diversity  # Add a small constant to avoid log(0)
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)  # Normalize probabilities to sum to 1
        
        # Ensure `np.random.choice` receives a valid probability distribution
        index = np.random.choice(len(prediction), p=prediction)
        
        # Ensure the index is valid (within range)
        if index >= len(int_to_note):
            index = random.randint(0, len(int_to_note) - 1)  # Fallback to a valid index
        
        result = int_to_note[index]
        
        # Ensure notes are in the major scale
        if result not in major_scale:
            result = major_scale[random.randint(0, len(major_scale) - 1)]
        
        generated_melody.append(result)
        
        # Update the input sequence
        sequence = np.append(sequence[0][1:], index)
        sequence = np.reshape(sequence, (1, len(sequence), 1))
    
    # Generate drums and bass alongside the melody
    drum_pattern = generate_drum_pattern(length=length)
    bass_line = generate_bass_line(generated_melody, major_scale, length=length)
    
    return generated_melody, drum_pattern, bass_line

# Convert the generated music, drums, and bass to MIDI
def convert_to_midi_with_drums_bass(melody, drums, bass, file_path='generated_song_with_drums_bass.mid'):
    midi_stream = stream.Stream()
    
    # Melody Part
    melody_part = stream.Part()
    melody_part.insert(0, instrument.Piano())  # Melody instrument
    
    for item in melody:
        if '.' in item:
            chord_notes = item.split('.')
            chord_notes = [note.Note(n) for n in chord_notes]
            melody_part.append(chord.Chord(chord_notes))
        else:
            melody_part.append(note.Note(item))
    
    midi_stream.append(melody_part)
    
    # Drums Part
    drum_part = stream.Part()
    drum_part.insert(0, instrument.Woodblock())  # Set to a percussion instrument
    
    for drum in drums:
        if drum == "Kick":
            drum_part.append(note.Rest(quarterLength=1.0))  # Simulate kick as a rest for simplicity
        elif drum == "Snare":
            drum_part.append(note.Note("C2", quarterLength=1.0))  # Simulate snare with a low note
        elif drum == "HiHat":
            drum_part.append(note.Note("C3", quarterLength=0.5))  # Simulate hi-hat with another note
    
    midi_stream.append(drum_part)
    
    # Bass Part
    bass_part = stream.Part()
    bass_part.insert(0, instrument.Bass())  # Bass instrument
    
    for bass_note in bass:
        bass_part.append(note.Note(bass_note, quarterLength=1.0))  # Bass note for each step
    
    midi_stream.append(bass_part)
    
    midi_stream.write('midi', fp=file_path)

# Generate and save the MIDI with melody, drums, and bass
midi_folder = r'E:\music generator\2015'  # Update this path
notes = load_midi_files(midi_folder)
sequences, next_notes, unique_notes, note_to_int, int_to_note = prepare_sequences(notes)

if len(sequences) == 0:
    raise ValueError("No sequences generated. Ensure that there are enough MIDI files and notes.")

input_shape = (sequences.shape[1], 1)  # Timesteps, features
output_dim = len(unique_notes)
model = build_cnn_gru_model(input_shape, output_dim)  # Build CNN-GRU model
train_cnn_gru_model(model, sequences, next_notes, epochs=0)  # Adjust epochs as needed

# Generate and save the MIDI file
midi_file_path = 'generated_song_with_drums_bass.mid'
generated_melody, drum_pattern, bass_line = generate_structured_music_with_drums_bass(model, sequences)
convert_to_midi_with_drums_bass(generated_melody, drum_pattern, bass_line, midi_file_path)

# Function to play MIDI file using pygame
def play_midi(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)

print(f"Playing MIDI file: {midi_file_path}")
play_midi(midi_file_path)
