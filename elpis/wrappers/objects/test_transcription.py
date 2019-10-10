import pytest
import json
from pathlib import Path

from elpis.wrappers.objects.interface import KaldiInterface

# from .test_pipeline import pipeline_upto_step_3

from . import test_pipeline


def test_new_transcription(tmpdir):
    """
    Check the state of a new transcription.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')
    assert t.has_been_transcribed == False
    assert t.exporter == None
    assert t.state == json.loads(f"""
    {{
        "name": "transcription_w",
        "hash": "{t.hash}",
        "model": null,
        "has_been_transcribed": false,
        "exporter": null
    }}
    """)
    return


def test_error_when_writing_to_protected_property(tmpdir):
    """
    An error is raised when there is an attempt to write to a protected
    property.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')

    with pytest.raises(NotImplementedError):
        t.has_been_transcribed = True
    with pytest.raises(NotImplementedError):
        t.exporter = "some obj"
    with pytest.raises(NotImplementedError):
        t.state = "Not a valid"
    with pytest.raises(NotImplementedError):
        t.model = "must link a model not assign it"


def test_new_transcription_using_override(tmpdir):
    """
    Using override has no effect when the pron dict with the same name does not
    exist.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w', override=True)
    assert t.has_been_transcribed == False
    assert t.exporter == None
    assert t.state == json.loads(f"""
    {{
        "name": "transcription_w",
        "hash": "{t.hash}",
        "model": null,
        "has_been_transcribed": false,
        "exporter": null
    }}
    """)
    return


def test_new_transcription_using_use_existing(tmpdir):
    """
    Using the use_existing when an existing transcription does not exist will
    produce a RuntimeError.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    with pytest.raises(RuntimeError):
        kaldi.new_transcription('transcription_w', use_existing=True)
    return


def test_existing_transcription_using_override(tmpdir):
    """
    Use override to delete a transcription with the same name and create a totally
    new transcription with the same name.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t1 = kaldi.new_transcription('transcription_w')
    t1_hash = t1.hash
    t2 = kaldi.new_transcription('transcription_w', override=True)
    # note t1 can no longer be used
    assert len(kaldi.list_transcriptions()) == 1
    assert t1_hash != t2.hash
    return


def test_two_new_transcription_with_same_name(tmpdir):
    """
    Trying to create two transcriptions with the same name without override or
    use_existing set to True will produce a ValueError.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    kaldi.new_transcription('transcription_w')
    with pytest.raises(ValueError):
        kaldi.new_transcription('transcription_w')
    return


def test_existing_transcription_using_use_existing(tmpdir):
    """
    Use the use_existing parameter to load configurations from existing pron dict.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t1 = kaldi.new_transcription('transcription_w')
    t1_hash = t1.hash
    t2 = kaldi.new_transcription('transcription_w', use_existing=True)
    assert len(kaldi.list_transcriptions()) == 1
    assert t1_hash != t2.hash
    assert t1.path == t2.path
    return


def test_override_and_use_existing(tmpdir):
    """
    Cannot have both the override and use_existing parameters set to True.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    with pytest.raises(ValueError):
        kaldi.new_transcription('transcription_w', override=True, use_existing=True)
    return


def test_linking(pipeline_upto_step_3):
    """
    Check state after linking.
    """
    kaldi, ds, pd, m = pipeline_upto_step_3
    t = kaldi.new_transcription('transcription_w')
    t.link(m)
    assert t.model == 'model_z'
    return


def test_transcribe_before_linking(tmpdir):
    """
    Generating an inference file before linking is not permitted and produces
    an error.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')
    with pytest.raises(RuntimeError):
        t.transcribe_to_text('/recordings/untranscribed/audio.wav')
    with pytest.raises(RuntimeError):
        t.transcribe_align('/recordings/untranscribed/audio.wav')
    return


def test_transcribe_to_text(pipeline_upto_step_3):
    """
    Test if some audio can be transcribed into plane text.
    """
    kaldi, ds, pd, m = pipeline_upto_step_3
    t = kaldi.new_transcription('transcription_w')
    t.link(m)
    text = t.transcribe_to_text('/recordings/untranscribed/audio.wav')
    assert text is str
    assert len(text) > 0
    # only transcribe_align chages this flag to True
    assert t.has_been_transcribed == False


def test_transcribe_align(pipeline_upto_step_3):
    """
    Test if some audio can be transcribed into some aligned utterances.
    """
    kaldi, ds, pd, m = pipeline_upto_step_3
    t = kaldi.new_transcription('transcription_w')
    t.link(m)
    t.transcribe_align('/recordings/untranscribed/audio.wav')

    # Black-box testing
    assert t.has_been_transcribed == True

    # White-box testing: check the inner content for CTM
    ctm_path = Path(f'{t.path}/align-words-best-wordkeys.ctm')
    assert ctm_path.is_file()
    return


def test_default_exporter(pipeline_upto_step_3):
    """
    After linking test if the default exporter is derived.
    """
    kaldi, ds, pd, m = pipeline_upto_step_3
    t = kaldi.new_transcription('transcription_w')
    assert t.exporter is None
    t.link(m)
    assert t.exporter is not None
    assert t.state['exporter']['name'] == 'Elan'
    return


def test_default_exporter_importer_only(tmpdir):
    """
    Sometimes a dataset will have a data transformer that is import only
    (exporting is disabled). Ensure the exporter in the transcription is
    nullified.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    ds = kaldi.new_dataset('dataset_x')
    ds.add_directory('/recordings/transcribed')
    ds.select_importer('Test_importer_import_only')
    ds.process()
    pd = kaldi.new_pron_dict('pron_dict_y')
    pd.link(ds)
    pd.set_l2s_path('/recordings/letter_to_sound.txt')
    pd.generate_lexicon()
    m = kaldi.new_model('model_z')
    m.link(ds, pd)
    m.build_kaldi_structure()
    m.train()
    t = kaldi.new_transcription('transcription_w')
    t.link(m)
    t.transcribe_align('/recordings/untranscribed/audio.wav')
    assert t.exporter is None
    # TODO: Fast way to test this?
    return


def test_set_exporter(tmpdir):
    """
    Set a new exporter.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')
    t.select_exporter('Elan')
    assert t.exporter is not None
    assert t.state['exporter']['name'] == 'Elan'
    return

def test_set_exporter_non_existant(tmpdir):
    """
    Setting an exporter that does not exist will raise an error.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')
    with pytest.raises(ValueError):
        t.select_exporter('Does not exist')
    return

def test_set_exporter_same_type(pipeline_upto_step_3):
    """
    Set a new exporter of the same type as the default. This should clear the
    context.
    """
    kaldi, ds, pd, m = pipeline_upto_step_3
    t = kaldi.new_transcription('transcription_w')
    t.exporter.change_tier('Shift')
    assert t.context['tire'] == 'Shift'
    t.select_exporter('Elan')
    assert t.context['tire'] == 'Phase'
    return


def test_set_exporter_with_importer_only(tmpdir):
    """
    Setting an exporter when it is an importer only will raise an error.
    """
    kaldi = KaldiInterface(f'{tmpdir}/state')
    t = kaldi.new_transcription('transcription_w')
    with pytest.raises(RuntimeError):
        t.select_exporter('Test_import_only')
    return

def test_use_exporter_by_content(pipeline_upto_step_4, tmpdir_factory):
    """
    Use an exporter to produce the transcription file content.
    """
    _,_,_,_, t = pipeline_upto_step_4
    t.select_exporter('Elan')
    t.transcribe_align()

    content = t.get_transcription_content()
    assert content is str
    assert len(content) != 0

    # test if the contents is a valid Elan File
    base_path = tmpdir_factory.mktemp("transcription_use_exporter_by_content")
    path_to_eaf = Path(f'{base_path}/transcription.eaf')
    with path_to_eaf.open(mode='wb') as fout:
        fout.write(content)
    _ = Eaf(f'{path_to_save}')
    return

def test_use_exporter_by_save(pipeline_upto_step_4, tmpdir_factory):
    """
    Use an exporter to save a transcription file.
    """
    _,_,_,_, t = pipeline_upto_step_4
    t.select_exporter('Elan')
    t.transcribe_align()

    base_path = tmpdir_factory.mktemp("transcription_use_exporter_by_save")
    path_to_save = Path(f'{base_path}/transcription.eaf')
    t.save_transcription_file(path_to_save)
    assert path_to_save.is_file()
    from pympi.Elan import Eaf
    _ = Eaf(f'{path_to_save}')
    return

def test_transcribe_with_untrained_model(pipeline_upto_step_0):
    """
    Attempting to transcribe with an untrained model will raise an error.
    """
    kaldi, = pipeline_upto_step_0
    m = kaldi.new_model('model_z')
    t.link(m)
    t.select_exporter('Elan')
    with pytest.raises(RuntimeError):
        t.transcribe_align()
    return
    

def test_exporter_before_transcribe_algin(pipeline_upto_step_4):
    """
    Attempting to export before transcribe_align is not possible and will raise
    an error.
    """
    _,_,_,_, t = pipeline_upto_step_4
    t.select_exporter('Elan')
    t.transcribe_align()
    with pytest.raises(RuntimeError):
        t.get_transcription_content()
    return
