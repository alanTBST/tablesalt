"""
Classes to read delrejser data
"""
from itertools import groupby
from operator import itemgetter
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread
import time
from typing import (
    Any,
    ClassVar,
    IO,
    Dict,
    Generator,
    Optional,
    List,
    Sequence,
    Tuple,
    Union,
    Set
    )
import zipfile

import pandas as pd  # type: ignore
from pandas._libs.parsers import TextReader #type: ignore
# from pandas.io.parsers import TextFileReader
from tablesalt.common.io import mappers #type: ignore
from tablesalt.preprocessing import TableArgParser #type: ignore
from datastores import make_store


MODEL_MAP = {v:k for k, v in mappers['model_dict'].items()}
CARD_MAP = mappers['card_id']


class InvalidPathContent(Exception):
    "error for content in path problems"
    pass

class FieldError(Exception):
    "error for missing field in delrejser"
    def __init__(self, column_name: str) -> None:
        self.column_name = column_name

class _DelrejserInspector:

    REQUIRED: ClassVar[Set[str]] = {
        'turngl', 'msgreportdate', 'stoppointnr', '', ''
        }

    def __init__(self, path: Path) -> None:
        """
        Check the contents of the delivered delrejser data

        :param path: a pathlib.Path object of a directory, zipfile or csv file
        :type path: Path
        :raises TypeError:  If the path argument isn't an instance of pathlib.Path
        :return: ''
        :rtype: None

        """

        self.path = path
        try:
            self.path_type = self._get_path_type()
        except AttributeError as e:
            raise TypeError("path arg must be an instance of pathlib.Path") from e

        self.content = self._get_content()
        self.headers = self._check_all_headers()

    def _get_path_type(self) -> str:
        """
        Get the type of path that is passed to __init__

        :raises InvalidPathContent: if the path doesn't have the right content
        :return: the type of path given
        :rtype: str

        """

        ptype: Optional[str] = None
        if self.path.is_dir():
            ptype = 'dir'

        suff = self.path.suffix
        if suff in ('.zip', '.csv'):
            ptype = suff
        if ptype is None:
            raise InvalidPathContent(
                "path can be one of [directory, .zip or .csv]"
                )
        return ptype

    @staticmethod
    def _get_sub_zips(lstzips: Sequence[Path]) -> Dict[Path, Tuple[str, ...]]:
        """get the names of the files in each zipfile"""
        return {zipf: tuple(zipfile.ZipFile(zipf).namelist()) for zipf in lstzips}

    @staticmethod
    def _expand_sub_zips(
            zips: Dict[Path, Tuple[str, ...]],
            fext: str
        ) -> List[Tuple[Path, str]]:

        all_files: List[Tuple[Path, str]] = []
        for k, v in zips.items():
            files = ((k, f) for f in v if fext if f)
            all_files.extend(files)

        return all_files

    def _get_content(self) -> List[str]:
        ptype = self.path_type

        fdict = {
            'dir': self._get_dir_content,
            '.zip': self._get_zip_content,
            '.csv': [self.path]
            }
        try:
            return fdict[ptype]()
        except TypeError:
            return fdict[ptype]

    def _get_dir_content(self) -> List[Tuple[Path, str]]:

        zips_in_path = list(self.path.glob('*.zip'))
        if not zips_in_path:
            raise FileNotFoundError(
                f"There are no zip files present in {self.path}"
                )
        zips = self._get_sub_zips(zips_in_path)
        all_files = self._expand_sub_zips(zips, '.csv')

        if not all_files:
            raise FileNotFoundError(
                "Could not find any csv files"
                )
        return all_files

    def _get_zip_content(self) -> List[Tuple[Path, str]]:

        zips = self._get_sub_zips([self.path])

        all_files = self._expand_sub_zips(zips, '.csv')

        return all_files
    @staticmethod
    def _get_headers(file: str) -> List[str]:
        try:
            df_0 = pd.read_csv(
                file, nrows=0, encoding='iso-8859-1'
                )
        except (FileNotFoundError, ValueError):
            file, content = file
            df_0 = pd.read_csv(
                zipfile.ZipFile(file).open(content),
                nrows=0,
                encoding='iso-8859-1'
                )

        file_columns = df_0.columns
        return [x.lower() for x in file_columns]

    def _check_all_headers(self) -> List[str]:
        """
        Determine if the data model in all files is the same
        :raises FieldError: If there is a difference
        :return: A list of the fields in the file
        :rtype: List[str]

        """

        first: List[str]
        first = self._get_headers(self.content[0])
        first_set = set(first)
        for c in self.content[1:]:
            cheaders = self._get_headers(c)
            diff = set(first_set) - set(cheaders)
            if diff:
                raise FieldError(
                    "All files must share the same data model. "
                    f"{c} contains fields "
                    f"{','.join(map(str, diff))} "
                    "that are not in not in the first file, "
                    "check all file column names"
                    )
        return first

class DataGenerator(_DelrejserInspector):
    """
    DataGenerator class to get data from files
    """

    def __init__(self, path: Path, *column: str) -> None:
        """
        A class for generating data from csv/zip files delivered by Rejsedata

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory
        :type path: Path
        :param *column: The columns of data to read.
        :type *column: str
        :return: ''
        :rtype: None

        """
        super().__init__(path)
        self.columns = {x.lower() for x in column}
        self.column_index = self._col_index_dict()


    def _reader(
            self,
            file_handle: Union[IO[bytes], IO[str]]
            ) -> pd._libs.parsers.TextReader:

        return TextReader(
            file_handle,
            encoding='iso-8859-1',
            usecols=[self.column_index[x] for x in self.columns]
            )

    def _col_index_dict(
            self
            ) -> Dict[str, int]:
        """
        Find the column index positions

        :return: A dictionary of column names and index positions
        :rtype: Dict[str, int]

        """

        try:
            kortnr = self.headers.index('kortnrkrypt')
        except ValueError:
            kortnr = self.headers.index('kortnr')

        colindices: Dict[str, int] = {}
        colindices['kortnr'] = kortnr

        for col in self.headers:
            try:
                colindices[col] = self.headers.index(col)
            except ValueError:
                pass
        return colindices

    def generate(self, chunksize: int = 500_000) -> Generator[Any, None, None]:
        """
        Create a generator of the given columns

        :param chunksize: The desired chunksize, defaults to 500_000
        :type chunksize: int, optional
        :yield: yields data that is specific to the child class.
        :rtype: Generator[Any, None, None]

        """

        for c in self.content:
            try:
                zipf, fp = c
                with zipfile.ZipFile(zipf).open(fp) as fh:
                    textgen = self._reader(fh)
                    while True:
                        try:
                            chunk = textgen.read(chunksize)
                            try:
                                yield self._process(chunk)
                            except AttributeError:
                                yield chunk
                        except StopIteration:
                            break

            except TypeError:
                textgen = self._reader(c.name)  # pathlib.Path.name
                while True:
                    try:
                        chunk = textgen.read(chunksize)
                        try:
                            yield self._process(chunk)
                        except AttributeError:
                            yield chunk
                    except StopIteration:
                        break

class TimeDataGenerator(DataGenerator):
    "TimeDataGenerator"
    def __init__(self, path: Path) -> None:
        """
        Generate data for the TimeStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.

        :type path: Path
        :return: ''
        :rtype: None

        """

        super().__init__(
            path,
            'turngl',
            'applicationtransactionsequencenu',
            'msgreportdate'
            )

    def _process(self, chunk):
        combined = zip(
            (int(x) for x in chunk[self.column_index['turngl']]),
            (int(x) for x in chunk[self.column_index['applicationtransactionsequencenu']]),
            (x for x in chunk[self.column_index['msgreportdate']]),
            )

        combined_sorted = sorted(combined, key=itemgetter(0, 1))

        return {key: {x[1]: x[2] for x in grp} for
                key, grp in groupby(combined_sorted, itemgetter(0))}

class TripUserGenerator(DataGenerator):
    """TripUserGenerator"""

    def __init__(self, path: Path) -> None:
        """
        Generate data for the TripUserStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.
        :type path: Path
        :return: ''
        :rtype: None

        """

        super().__init__(path, 'turngl', 'kortnrkrypt')

    def _process(self, chunk):

        return dict(zip(
                chunk[self.column_index['turngl']],
                chunk[self.column_index['kortnrkrypt']]))

class StopDataGenerator(DataGenerator):

    def __init__(self, path: Path) -> None:
        """
        Generate data for the StopStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.
        :type path: Path
        :return: ''
        :rtype: None

        """
        super().__init__(
            path, 'turngl',
            'applicationtransactionsequencenu',
            'stoppointnr', 'model'
            )

    def _process(self, chunk):
        "stop specific processing"

        combined = zip(
            (int(x) for x in chunk[self.column_index['turngl']]),
            (int(x) for x in chunk[self.column_index['applicationtransactionsequencenu']]),
            (int(x) for x in chunk[self.column_index['stoppointnr']]),
            (MODEL_MAP[x] for x in chunk[self.column_index['model']])
            )

        combined_sorted = sorted(combined, key=itemgetter(0, 1))

        return {key: {x[1]: {'stop': x[2], 'model': x[3]} for x in grp} for
                key, grp in groupby(combined_sorted, key=itemgetter(0))}


class PassengerDataGenerator(DataGenerator):

    def __init__(self, path: Path) -> None:
        """
        Generate data for the PassengerStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.
        :type path: Path
        :return: ''
        :rtype: None

        """

        super().__init__(
            path, 'turngl', 'passagerantal1',
            'passagerantal2', 'passagerantal3',
            'passagertype1', 'passagertype2',
            'passagertype3', 'korttype'
            )
    @staticmethod
    def _pre_process(seq: Sequence):
        "deal with empty ptype data"
        return [x if x != '' else 0 for x in seq]

    def _process(self, chunk):
        "passenger specific processing"

        combined = zip(
            (int(x) for x in chunk[self.column_index['turngl']]),
            (int(x) for x in chunk[self.column_index['passagerantal1']]),
            (int(x) for x in chunk[self.column_index['passagerantal2']]),
            (int(x) for x in chunk[self.column_index['passagerantal3']]),
            (int(x) for x in self._pre_process(chunk[self.column_index['passagertype1']])),
            (int(x) for x in self._pre_process(chunk[self.column_index['passagertype2']])),
            (int(x) for x in self._pre_process(chunk[self.column_index['passagertype3']])),
            (CARD_MAP[x] for x in chunk[self.column_index['korttype']]),
            )
        combined_sorted = sorted(combined, key=itemgetter(0))
        combined_filtered = [
            x for x in combined_sorted if any(y > 0 for y in x[1:7])
            ]

        out = {
            x[0]: {
                'p1': x[1], 't1': x[4],
                'p2': x[2], 't2': x[5],
                'p3': x[3], 't3': x[6],
                'pt': sum(x[1:4]), 'c': x[7]
                } for x in combined_filtered
            }

        return {k: {k1: v1 for k1, v1 in v.items() if v1 > 0} for
                k, v in out.items()}

class PriceDataGenerator(DataGenerator):

    def __init__(self, path: Path) -> None:
        """
        Generate data for the PriceStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.
        :type path: Path
        :return: ''
        :rtype: None

        """

        super().__init__(
            path,
            'turngl',
            'rejsepris',
            'tidsrabat',
            'zonerrejst'
            )

    def _process(self, chunk):
        """price specific processing"""
        combined = zip(
            chunk[self.column_index['turngl']],
            chunk[self.column_index['rejsepris']],
            chunk[self.column_index['tidsrabat']],
            chunk[self.column_index['zonerrejst']]
            )
        combined = (x for x in combined if x[3] > 0)

        return {x[0]: {
            'price': float(x[1].replace('.', '').replace(',', '.')),
            'rabat': float(x[2].replace('.', '').replace(',', '.')),
            'zones': int(x[3])
            } for x in combined}

class OperatorGenerator(DataGenerator):

    def __init__(self, path: Path) -> None:
        """
        Generate data for the operatorStore

        :param path: The input path of the data. Can be the path to a csv file,
            a zip file or a path to a directory of zipfiles.
        :type path: Path
        :return: ''
        :rtype: None

        """

        super().__init__(
            path,
            'turngl',
            'applicationtransactionsequencenu',
            'nyudfører',
            'contractorid',
            'ruteid',
            )

    def _process(self, chunk):
        combined = zip(
            chunk[self.column_index['turngl']],
            chunk[self.column_index['applicationtransactionsequencenu']],
            chunk[self.column_index['nyudfører']],
            chunk[self.column_index['contractorid']],
            chunk[self.column_index['ruteid']]
            )
        combined_sorted = sorted(combined, key=itemgetter(0, 1))

        return {
            int(key): {
                int(x[1]): {
                    'operator': x[2], 'contractor': x[3], 'route': x[4]
                    } for x in grp
                }
            for key, grp in groupby(combined_sorted, key=itemgetter(0))
            }



def blocks(files: IO[bytes]) -> Generator[bytes, None, None]:
    """
    Yield bytes of the file until the endo of the file is reached

    :param files: a file of io oobject
    :type files: IO[bytes]
    :yield: 65536 bytes of the file
    :rtype: Generator[bytes, None, None]

    """

    while True:
        b = files.read(65536)
        if not b:
            break
        yield b


def sumblocks(zfile: IO[bytes], content: Union[str, zipfile.ZipInfo]) -> int:
    """
    Get the total number of lines in the zipfile content

    :param zfile: a path to a zipfile
    :type zfile: IO[bytes]
    :param content: the subfile contents of the given zfile
    :type content: Union[str, zipfile.ZipInfo]
    :return: the number of lines in the file
    :rtype: int

    """

    with zipfile.ZipFile(zfile).open(content) as f:
        n_lines = sum(
            bl.count(b"\n") for bl in blocks(f)
            )
    return n_lines

def check_collection_complete(arr_col, key):
    """cheack the common.io collection mapper is the same"""
    test_vals = set(mappers[key].keys())
    unseen = set()

    for x in arr_col:
        if x not in test_vals:
            unseen.add(x)
    if not unseen:
        return None
    if any(isinstance(x, int) for x in unseen):
        return None
    return unseen

# def update_collection(unseen_ids, key):
#     """change the io mappers if needed"""
#     # TODO put this updating activity in a class
#     current_max_key_id = max(mappers[key].values())

#     package_loc = site.getsitepackages()
#     collection_loc = os.path.join(
#         package_loc[1], 'tablesalt', 'common',
#         'io', 'rejsekortcollections.json'
#         )
#     try:
#         with open(collection_loc, 'r', encoding='iso-8859-1') as f:
#             old_collection = json.loads(f.read().encode('iso-8859-1').decode())

#         for i, x in enumerate(unseen_ids):
#             mappers[key][x] = current_max_key_id + 1 + i
#             old_collection[key][x] = current_max_key_id + 1 + i

#         with open(collection_loc, 'w') as fp:
#             json.dump(old_collection, fp)
#     except Exception as e:
#         print(str(e))
#         print("\n")
#         print("skipping mappers update")


def _determine_path(d) -> str:
    """get the path of the input dictionary from structure of
    the dictionary input """
    for _, v in d.items():
        break
    try:
        _ = v
    except NameError as e:
        raise ValueError from e

    if isinstance(v, str):
        return 'tripcard'
    if 'price' in v:
        return 'price'
    if 'p1' in v:
        return 'pas'

    sample = list(v.values())[0]

    if 'stop' in sample:
        return 'stops'
    if 'operator' in sample:
        return 'operator'
    return 'time'

def _data_producer(data_generator: DataGenerator, queue) -> None:
    """
    Create data from the DataGenerator on put it in a queue to be written

    :param data_generator: the data generator to produce
    :type data_generator: DataGenerator
    :param queue: the queue to put data in
    :type queue: multiprocessing.Queue
    :return: ''
    :rtype: None

    """


    generator = data_generator.generate(50_000)
    for x in generator:
        queue.put(x)
        time.sleep(0.01)


def _data_consumer(queue, base_path: str) -> None:
    """
    Take data from the queue and write it to an lmdb store

    :param queue: the queue to get data from
    :type queue: multiprocessing.Queue
    :param base_path: the output path for the delrejser store
    :type base_path: str
    :return: ''
    :rtype: None

    """


    while True:
        if not queue.empty():
            res = queue.get()
            if res is None:
                break
        else:
            time.sleep(0.02)
            continue
        path = _determine_path(res)
        write_path = Path(base_path) / path
        make_store(res, str(write_path))

def delrejser_setup(input_path: str, output_path: str) -> None:
    """
    Setup all of the lmdb key-value stores

    :param input_path: the path containing the delrejser data
    :type input_path: str
    :param output_path: DESCRIPTION
    :type output_path: str
    :return: DESCRIPTION
    :rtype: None

    """

    generators = [
        StopDataGenerator,
        TimeDataGenerator,
        PriceDataGenerator,
        PassengerDataGenerator,
        OperatorGenerator,
        TripUserGenerator
        ]

    the_queue = Queue()
    # producers
    producers = []
    for gen in generators:
        gener = gen(input_path)
        p = Process(target=_data_producer, args=(gener, the_queue))
        producers.append(p)

    st = time.time()
    # consumer
    t1 = Thread(target=_data_consumer, args=(the_queue, output_path))

    for p in producers:
        p.start()

    t1.start()

    for p in producers:
        p.join()
    the_queue.put(None)
    t1.join()
    et = time.time()
    print((et-st) / 60)


if __name__ == "__main__":
    parser = TableArgParser('input_dir', 'output_dir')
    args = parser.parse()
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    delrejser_setup(input_dir, output_dir)
