import json
import time
from abc import ABC, abstractclassmethod, abstractmethod
from pathlib import Path
from elpis.wrappers.utilities import hasher

# Design constraint
# Since there are four classes that must have their states saved to the
# operating system, this single class was made to provide some common
# functionality and a standard of operation for these classes. The alternative
# to using this method of storing everything on disk (using the file system
# directly) is to implement a database, however, kaldi requires access to
# files and a specific file structure. This was the constrain that lead to the
# FSObject.


# The classes that use FSObject as a base are: Dataset, Model, Transcription
# and KaldiInterface.


class FSObject(ABC):
    def __init__(self,
                 parent_path: Path = None,
                 dir_name: str = None,
                 name: str = None,
                 pre_allocated_hash: str = None
                 ):
        # Not allowed to instantiate this base class
        if type(self) == FSObject:
            raise NotImplementedError('Must inherit FSObject, not instantiate it.')

        # Must have a _config_file variable
        self._config_file
        
        # _config_file must be a JSON file
        if not self._config_file.endswith('.json'):
            raise ValueError('_config_file must be a JSON file (ends with ".json")')

        # Optional arg: pre_allocated_hash
        if pre_allocated_hash is None:
            h = hasher.new()
        else:
            h = pre_allocated_hash

        # Optional arg: dir_name
        if dir_name is None:
            dir_name = h

        # path to the object
        self.__path = Path(parent_path).joinpath(dir_name)
        self.path.mkdir(parents=True, exist_ok=True)
        #  if no config, then create it
        config_file_path = Path(f'{self.__path}/{self._config_file}')
        if not config_file_path.exists():
            self.ConfigurationInterface(self)._save({})
        self.config['name'] = name
        self.config['hash'] = h
        self.config['date'] = str(time.time())

    def _initial_config(self, config):
        self.ConfigurationInterface(self)._save(config)

    

    @classmethod
    def load(cls, base_path: Path):
        """
        Create the proxy FSObject from an existing one in the file-system.

        :param base_path: is the path to the FSObject representation.
        :return: an instansiated FSObject proxy.
        """
        self = cls.__new__(cls)
        self.__path = Path(base_path)
        return self
    
    @property
    @abstractmethod
    def _config_file(self) -> str:
        raise NotImplementedError('no _config_file has been defined for this class')
        return 'NotImplemented'

    @property
    def path(self) -> Path:
        """write protection on self.path"""
        return Path(self.__path)

    @property
    def name(self) -> str:
        return self.config['name']

    @name.setter
    def name(self, value: str):
        self.config['name'] = value

    @property
    def hash(self) -> str:
        return self.config['hash']

    def __hash__(self) -> int:
        return int(f'0x{self.hash}', 0)

    @property
    def date(self):
        return self.config['date']

    @property
    def config(self):
        return self.ConfigurationInterface(self)

    class ConfigurationInterface(object):
        """
        Continuesly save changes to disk and only read properties from disk
        (in the JSON file storing the objects configuration).

        This class is more syntax sugar. Particularly so we can treat the
        'config' attribute/property in the FSObject class like a JSON
        (or dict), since it is interfacing directly with one.
        """
        def __init__(self, fsobj):
            self.fsobj = fsobj

        def _file_name(self):
            return getattr(self.fsobj, '_config_file', 'config.json')

        def _load(self):
            with open(f'{self.fsobj.path}/{self._file_name()}', 'r') as fin:
                return json.load(fin)

        def _save(self, conf):
            with open(f'{self.fsobj.path}/{self._file_name()}', 'w') as fout:
                return json.dump(conf, fout)

        def __getitem__(self, key: str):
            return self._load()[key]

        def __setitem__(self, key, value):
            config = self._load()
            config[key] = value
            self._save(config)

        def __repr__(self):
            return self._load().__repr__()

        def __str__(self):
            return self._load().__str__()
