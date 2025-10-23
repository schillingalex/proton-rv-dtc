from dataclasses import dataclass
from jsons import JsonSerializable

from util.import_utils import instance_from_string


class ExtendedJsonSerializable(JsonSerializable):
    """
    Used to extend a jsons serializable class with additional convenience methods for reading and writing files.
    """

    @classmethod
    def from_file(cls, path: str):
        """
        Loads a JSON file from the given path and deserializes it using jsons.

        :param path: Path to the target JSON file to load.
        :return: An ExtendedJsonSerializable instance of the calling subclass.
        """
        with open(path, "r") as f:
            return cls.loads(f.read())

    def to_file(self, path: str):
        """
        Serializes an ExtendedJsonSerializable instance to a JSON file at the specified path.

        :param path: Path to the target JSON file to write.
        """
        with open(path, "w") as f:
            f.write(self.dumps())


@dataclass
class InstantiableJsonSerializable(ExtendedJsonSerializable):
    classname: str
    args: dict

    def new_instance(self, **kwargs):
        instance = instance_from_string(self.classname, **self.args, **kwargs)
        return instance
