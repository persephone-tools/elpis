from pathlib import Path
import threading
from typing import Callable

from elpis.wrappers.objects.dataset import Dataset
from elpis.wrappers.objects.fsobject import FSObject
from elpis.wrappers.objects.pron_dict import PronDict

class PersephoneModel(FSObject):
    _config_file = "model.json"

    # NOTE I'm essentially copying a lot of the framework in
    # wrappers/objects/model.py, so perhaps there should be a KaldiModel and
    # PersephoneModel subclass instead.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset: Dataset = None
        self.config["dataset_name"] = None
        self.config["status"] = "untrained"
        self.status = "untrained"

    @classmethod
    def load(cls, base_path: Path):
        self = super().load(base_path)
        self.dataset = None
        self.pron_dict = None
        return self

    @property
    def status(self):
        return self.config["status"]

    @status.setter
    def status(self, value: str):
        self.config["status"] = value

    def link(self, dataset: Dataset, _pron_dict: PronDict):
        # NOTE Persephone doesn't do anything with the pron dict.
        self.dataset = dataset
        self.config["dataset_name"] = dataset.name

    @property
    def ngram(self) -> int:
        return None

    def build_kaldi_structure(self):
        # TODO Rename this function because it has nothing to do with Kaldi.

        # Convert the JSON into Persephone-compatible format. The most basic
        # option here is to just split the WAVs up and create a directory
        # compatible with the current Persephone requirements. Another option
        # would be to make Persephone support the Elpis JSON. Yet another option
        # would be to make Persephone support the Kaldi format!
        # I might go with the first option for now since it's probably the
        # easiest. Making Persephone Kaldi-compatible would probably be the
        # best long-term route.

        wav_dir = self.path / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        label_dir = self.path / "label"
        label_dir.mkdir(parents=True, exist_ok=True)

        # Load the transcriptions
        transcription_f = self.dataset.pathto.filtered_json
        import json
        with open(transcription_f) as f:
            transcriptions = json.load(f)
        from persephone.preprocess.wav import trim_wav_ms
        for utter_id, transcription in enumerate(transcriptions):
            audio_path = self.dataset.path / "resampled" / transcription["audio_file_name"]
            transcript = transcription["transcript"]
            start_ms = transcription["start_ms"]
            stop_ms = transcription["stop_ms"]
            speaker_id = transcription["speaker_id"]

            trimmed_wav_path = wav_dir / "{}.wav".format(utter_id)
            trim_wav_ms(audio_path, trimmed_wav_path,
                        start_ms, stop_ms)
            # TODO hardcoding chars here, but should support other
            # segmentations
            label_path = label_dir / "{}.chars".format(utter_id)
            with open(label_path, "w") as f:
                no_spaces = "".join(transcript.split())
                chars = " ".join(no_spaces)
                print(chars, file=f)
        from persephone.corpus import Corpus
        self.corpus = Corpus("fbank", "chars", self.path)

    def train(self, on_complete:Callable=None):
        self.status = "training"
        from persephone.corpus_reader import CorpusReader
        reader = CorpusReader(self.corpus, batch_size=1)
        from persephone import rnn_ctc
        exp_dir = self.path / "exp" / "test_exp"
        model = rnn_ctc.Model(exp_dir, reader, num_layers=2, hidden_size=250)

        # TODO for now setting this as None so that training logs get output to
        # flask log
        on_complete = None

        def run_training_in_background():
            def background_train_task():
                model.train()
                self.status = "trained"
                on_complete()
            t = threading.Thread(target=background_train_task)
            t.start()

        if on_complete:
            run_training_in_background()
        else:
            model.train(max_epochs=10)
            self.status = "trained"
