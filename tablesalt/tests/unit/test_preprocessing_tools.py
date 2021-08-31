

from tablesalt.preprocessing.tools import (check_all_file_headers,
                                           col_index_dict, get_columns,
                                           get_zips, setup_directories,
                                           sumblocks)


def test_get_zips():
    assert False


def test_setup_directories(lib_dir):

    assert not lib_dir.is_dir()
    #with monkeypatch.context() as m:
       # m.setattr(weather, 'DEFAULT_SAVE_PATH', os.path.join(lib_dir, 'data.json'))
    # Library should exist after save.
    #assert lib_dir.is_dir()
    assert (lib_dir / 'data.json').is_file()


def test_get_columns():

    assert False

def check_all_file_headers():

    assert False

