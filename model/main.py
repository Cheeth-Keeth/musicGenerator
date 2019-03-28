import glob
import pickle

import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils
from music21 import converter, instrument, note, chord


def main():
    notes = get_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequence(notes, n_vocab)

    print(network_input.shape)

    model = create_network(network_input, n_vocab)

    print(model.summary())

    train(model, network_input, network_output)


def get_notes():
    notes = []
    count = 0
    for file in glob.glob('model/piano/*.midi'):
        count += 1
        midi = converter.parse(file)
        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for part in notes_to_parse:
            if isinstance(part, note.Note):
                notes.append(str(part.pitch))
            elif isinstance(part, chord.Chord):
                notes.append('.'.join(str(n) for n in part.normalOrder))

    with open('model/data/notespiano', 'wb') as filepath:
        pickle.dump(notes, filepath)

    print(notes)
    return notes


def prepare_sequence(notes, n_vocab):
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    # hidden_size = 512
    # model = Sequential()
    # model.add(LSTM(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     return_sequences=True
    # ))
    # model.add(LSTM(hidden_size, return_sequences=True))
    # model.add(Dense(256))
    # model.add(TimeDistributed(Dense(n_vocab)))
    # model.add(Activation('softmax'))
    # optimizer = Adam()
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # return model

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    checkpoint = ModelCheckpoint(filepath='model/data' + '/piano-{epoch:02d}.hdf5', monitor='loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':
    main()
