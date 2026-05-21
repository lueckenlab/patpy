from pathlib import Path


class PatpyConfig:
    """Global configuration for patpy.

    Currently exposes a single ``datasetdir`` knob that controls where the
    dataset loaders in :mod:`patpy.datasets` cache their downloaded files.
    Mirrors the role that ``scanpy.settings.datasetdir`` plays for scanpy.
    """

    def __init__(self, datasetdir: str | Path = "data"):
        self._datasetdir = Path(datasetdir)

    @property
    def datasetdir(self) -> Path:
        """Directory used to cache downloaded datasets."""
        return self._datasetdir

    @datasetdir.setter
    def datasetdir(self, value: str | Path) -> None:
        self._datasetdir = Path(value)

    def __repr__(self) -> str:
        return f"PatpyConfig(datasetdir={self._datasetdir!r})"


settings = PatpyConfig()
