import pickle
from collections import Counter
import re

from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed, Activation
from keras.models import Sequential, load_model
# from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

import numpy as np


def read_text(path, chapter_byminlength=10, trim=True, chapter='---', extra_paths=tuple()):
    paths = (path,) + extra_paths
    content = []
    for path in paths:
        with open(path, 'r') as f:
            this_content = f.readlines()
        if trim:
            this_content = list(trim_line_start(this_content))
        content += list(separate_puntuation(this_content))
    def generator():
        for line in content:
            line = line.strip()
            if chapter_byminlength is not None and len(line) < chapter_byminlength:
                yield
            elif line == chapter:
                yield
            elif line:
                for word in line.split():
                    if word.strip():
                        yield word.strip()
    return generator


def separate_puntuation(content):
    pattern = re.compile(r'([.,:;!?])')
    pattern2 = re.compile(r'(["\(\)])')
    for line in content:
        yield pattern2.sub(r' \1 ', pattern.sub(r' \1', line))


def trim_line_start(content):
    m = re.compile(r'\[\d+\] ?(.*)')
    m2 = re.compile(r'\d+[.:] ?(.*)')
    for line in content:
        match = m.match(line)
        if match:
            yield match.group(1)
        match = m2.match(line)
        if match:
            yield match.group(1)
        else:
            yield line


def get_dictionaries(generator):
    count = Counter(generator()).most_common()
    dictionary = {}
    rev_dictionary = {}
    sub = 0
    for idx, (word, _) in enumerate(count):
        if word is None:
            sub = 1
            continue
        dictionary[word] = idx - sub
        rev_dictionary[idx - sub] = word
    return dictionary, rev_dictionary


def to_chapters(generator, dictionary):
    phrase = []
    for word in generator():
        if word is None and phrase:
            yield phrase
            phrase = []
        elif word is not None:
            phrase.append(dictionary[word])
    if phrase:
        yield phrase


def to_train_x_y(input_generator, sequence_length=5, predict_words=1):
    for line in input_generator:
        for idx in range(len(line) - (sequence_length + predict_words - 1)):
            yield (
                line[idx: idx + sequence_length],
                line[idx + predict_words: idx + predict_words + sequence_length]
            )



def get_model(num_words, nodes=(50, 100, 250), sequence_length=5):
    model = Sequential()
    model.add(Embedding(num_words, nodes[0], input_length=sequence_length))
    model.add(LSTM(nodes[1], dropout=.2, recurrent_dropout=.2, return_sequences=True))
    model.add(LSTM(nodes[2], activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_words)))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class PredictiveText:
    def __init__(
        self, model, dictionary, reverse_dictionary, training_data,
        training_data_y=None, batch_size=None, predict_words=1,
    ):
        self._training_data = training_data
        self._training_data_y = training_data_y
        self.batch_size = batch_size
        self.model = model
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.sentence_punctuation = '!.?'
        self.predict_words = predict_words

    def to_text(self, sequence):
        phrase = ' '.join([self.reverse_dictionary[word] for word in sequence])
        pattern = re.compile(r' ([.,:;!?])')
        return pattern.sub(r'\1', phrase)

    def get_random_seed(self, sentence_start=True):
        starts = []
        include_next = False
        for sequenceidx, sequence in enumerate(self._training_data):
            if sentence_start:
                if sequenceidx == 0 or include_next:
                    starts.append(sequenceidx)
                    include_next = False
                elif self.reverse_dictionary[sequence[-1]] in self.sentence_punctuation:
                    include_next = True

            else:
                starts.append(sequenceidx)
        sequenceidx = np.random.randint(len(starts))
        return self._training_data[sequenceidx]

    def train(self, epochs=10, eons=42, speaks_each_eon=3):

        if self._training_data_y is None:
            print("Sorry no y data, can't train further")
            return

        for _ in range(eons):
            self.model.fit(
                np.array(self._training_data),
                self._training_data_y,
                batch_size=self.batch_size,
                epochs=epochs,
            )
            for __ in range(speaks_each_eon):
                print("---")
                print(self.speak(careful=True))

    def speak(self, seed=None, maxlen=300, max_sentences=5, careful=False):
        """Produce text.

        Arguments:
            seed
                a sequence of numbers representing words to get it going
                if blank, a random seed from the training text will be
                used.
            maxlen
                How many words the output may be
            max_sentences
                How many sentences the output may be
            careful
                If True will use the part of the seed that doesn't
                overlap for next iteration's seed

        Returns:
            A generated text
        """
        if seed is None:
            seed = self.get_random_seed()
        sequence_length = len(seed)
        phrase = [w for w in seed]
        nextword = seed[-1]
        sentences = 0
        assert len(seed[-self.predict_words:]) == self.predict_words
        while sentences < max_sentences and len(phrase) < maxlen:
            nextsequence = np.argmax(
                self.model.predict(np.array(seed)[np.newaxis, ...])[0],
                axis=1,
            ).tolist()
            phrase.extend(nextsequence[-self.predict_words:])
            for word in nextsequence[-self.predict_words:]:
                if self.reverse_dictionary[word] in '!.?':
                    sentences += 1
            if careful:
                seed = seed + nextsequence
                seed = seed[-sequence_length:]
            else:
                seed = nextsequence
        return self.to_text(phrase)

    def save(self, path):
        self.model.save(path)
        with open(path + '.params', 'wb') as f:
            pickle.dump({
                'training_data': self._training_data,
                'training_data_y': self._training_data_y,
                'batch_size': self.batch_size,
                'dictionary': self.dictionary,
                'reverse_dictionary': self.reverse_dictionary,
                'predict_words': self.predict_words,
            }, f)

    @staticmethod
    def load(path):
        model = load_model(path)
        with open(path + '.params', 'rb') as f:
            extra = pickle.load(f)
        return PredictiveText(model, **extra)


def train(
    path, *extra_paths, chapter_byminlength=10, model=None,
    batch_size=64, epochs=30, sequence_length=5, predict_words=1,
):
    """Train based on text.

    One or more paths to text files.

    Keyword arguments:

        chapter_byminlength:
            If a line is less than this many characters
            it is a chapter heading. Going from one chapter
            to the next is not learnt.
            chapters can also be separated by a line
            containing only `---`.
        model:
            If you want to continue training on a previous
            model (`result.model`).
        batch_size:
            Number of sequences trained each time
        epochs:
            How long it will train for
        sequence_length:
            How many words are given to guess what the
            next word is.
        predict_words:
            Number of words to predict, max is the sequence
            length

    Returns:
        A predictive text result (`PredictiveText`)

    """
    predict_words = min(sequence_length, predict_words)
    content = read_text(
        path, chapter_byminlength=chapter_byminlength, extra_paths=extra_paths,
    )
    dictionary, reverse_dictionary = get_dictionaries(content)
    nwords = len(dictionary)
    print("{} different words".format(nwords))
    chapters = list(to_chapters(content, dictionary))
    print("{} chapters or length {} - {} words".format(
        len(chapters),
        min(len(chapter) for chapter in chapters),
        max(len(chapter) for chapter in chapters),
    ))
    x, y = zip(*to_train_x_y(
        chapters, sequence_length, predict_words=predict_words,
    ))
    y = to_categorical(np.array(y), nwords)
    if model is None:
        model = get_model(nwords, sequence_length=sequence_length)
    else:
        print("Continuing training previous model")
    pred_text = PredictiveText(
        model, dictionary, reverse_dictionary, x, y, batch_size,
        predict_words=predict_words,
    )
    pred_text.train(epochs=epochs, eons=1)
    return pred_text
