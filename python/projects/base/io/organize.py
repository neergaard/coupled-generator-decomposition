import copy
from dataclasses import dataclass
from pathlib import Path
import pprint


class Organizer(object):
    def __init__(self, attrs, special_attrs=None):
        self.special_attrs = special_attrs
        self.attr2abbr = {}
        # Set the attributes
        for attr in attrs:
            setattr(self, attr, None)
            self.attr2abbr[attr] = attr
        # Set the special attributes
        for attr in self.special_attrs:
            if not hasattr(self, attr):
                setattr(self, attr, None)
            if attr in self.attr2abbr:
                del self.attr2abbr[attr]

    def set_attr2abbr(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                self.attr2abbr[attr] = value

    def clear_attrs(self, keep=None):
        add_keep = ["attr2abbr", "special_attrs"]
        keep = list(keep) + add_keep if keep is not None else add_keep
        for attr in self.__dict__:
            if keep is None or attr not in keep:
                setattr(self, attr, None)

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            # assert attr in self.__dict__.keys(), 'Invalid attribute!'
            if attr in self.__dict__:
                setattr(self, attr, value)

    def prepend(self, **kwargs):
        for attr, value in kwargs.items():
            # assert attr in self.__dict__.keys(), 'Invalid attribute!'
            if attr in self.__dict__:
                setattr(self, attr, value + getattr(self, attr))

    def append(self, **kwargs):
        for attr, value in kwargs.items():
            # assert attr in self.__dict__.keys(), 'Invalid attribute!'
            if attr in self.__dict__:
                setattr(self, attr, getattr(self, attr) + value)

    def _choose(self, attr, kwargs=None):
        if kwargs is not None and attr in kwargs:
            return kwargs[attr]
        else:
            return getattr(self, attr)

    def __repr__(self):
        return pprint.pformat(self.__dict__)


class PathOrganizer(Organizer):
    def __init__(self, attrs):
        special_attrs = ("root",)
        super().__init__(attrs, special_attrs)

    def get(self, **kwargs):
        """Get the path."""
        assert isinstance(self.root, Path), "root must be a Path object!"
        path = self.root
        for attr, abbr in self.attr2abbr.items():
            if value := self._choose(attr, kwargs):
                if abbr not in (None, ""):
                    value = abbr + "-" + value
                path = path / value
        return path

    def ensure_exists(self):
        path = self.get()
        if not path.exists():
            path.mkdir(parents=True)


class FileOrganizer(Organizer):
    def __init__(self, attrs):
        special_attrs = ("prefix", "suffix", "extension")
        super().__init__(attrs, special_attrs)

    def get_stem(self, **kwargs):
        """Get the path."""
        stem = ""
        if prefix := self._choose("prefix", kwargs):
            stem += prefix
        for attr, abbr in self.attr2abbr.items():
            if value := self._choose(attr, kwargs):
                stem += f"_{abbr}-{value}"
        if suffix := self._choose("suffix", kwargs):
            stem += "_" + suffix
        stem = stem.lstrip("_")
        return stem

    def get(self, **kwargs):
        name = self.get_stem(**kwargs)
        if ext := self._choose("extension", kwargs):
            name += "." + ext
        return name


class FileFormatter(object):
    def __init__(self, path_attrs, file_attrs):
        self.path = PathOrganizer(path_attrs)
        self.file = FileOrganizer(file_attrs)

    def set_attr2abbr(self, **kwargs):
        self.path.set_attr2abbr(**kwargs)
        self.file.set_attr2abbr(**kwargs)

    def update(self, **kwargs):
        self.path.update(**kwargs)
        self.file.update(**kwargs)

    def prepend(self, **kwargs):
        self.path.prepend(**kwargs)
        self.file.prepend(**kwargs)

    def append(self, **kwargs):
        self.path.append(**kwargs)
        self.file.append(**kwargs)

    def clear_attrs(self, keep=None):
        self.path.clear_attrs(keep)
        self.file.clear_attrs(keep)

    def get_filename(self, **kwargs):
        return self.path.get(**kwargs) / self.file.get(**kwargs)

    def get_state(self):
        return copy.deepdopy(self.__dict__)

    def set_state(self, state):
        self.__dict__ = state

    def __repr__(self):
        s = "PATH\n"
        s += "----------\n"
        s += pprint.pformat(self.path.__dict__)
        s += "\n\n"
        s += "FILE\n"
        s += "----------\n"
        s += pprint.pformat(self.file.__dict__)
        return s


@dataclass
class SimnibsOrganizer(object):
    root: Path
    subject: str = None

    def update(self, **kwargs):
        for attr, value in kwargs.items():
            assert attr in self.__dict__.keys(), "Invalid attribute!"
            setattr(self, attr, value)

    def get_path(self, rel_path):
        path = self.root / f"sub-{self.subject}"
        if rel_path != "subject":
            path = path / f"{rel_path}_sub-{self.subject}"
        return path

    def match(self, rel_path, from_path="m2m"):
        assert self.subject is not None

        path = self.get_path(from_path)

        match = tuple(path.glob(rel_path))
        assert len(match) > 0, f"Found no match for {path / rel_path}"
        if len(match) == 1:
            match = match[0]

        return match
