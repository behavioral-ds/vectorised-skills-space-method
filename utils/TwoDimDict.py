from dataclasses import dataclass
from typing import Any


@dataclass
class TwoDimDict:
    dictionary = {}

    def add_key_val_pair(self, key1: str | int, key2: str | int, value: Any):
        if key2 in self.dictionary and key1 in self.dictionary[key2]:
            self.dictionary[key2][key1] = value
            return

        if not key1 in self.dictionary:
            self.dictionary[key1] = {}

        self.dictionary[key1][key2] = value

    def get_value(self, key1: str | int, key2: str | int) -> Any:
        if key1 in self.dictionary and key2 in self.dictionary[key1]:
            return self.dictionary[key1][key2]

        if key2 in self.dictionary and key1 in self.dictionary[key2]:
            return self.dictionary[key2][key1]

        return None
