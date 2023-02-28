from pathlib import Path

import pandas as pd
from loguru import logger


class Load:
    def __init__(self, data_dir: str = "../../data"):
        """Initialize the Load class with the data directory to your data.
        For example, if you want to load the data for the `tweets`, the `data_dir` should be `../../data/tweets`.
        This parameter severs the purpose to reduce the length when loading an individual file.

        Parameters
        ----------
        data_dir : str, optional
            The directory where your main data is stored. By default `../../data`.
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            logger.error(f"The data directory {self.data_dir} does not exist.")
            raise FileNotFoundError(f"The data directory {self.data_dir} does not exist.")

    def load_dataframe(self, file_path: str, columns: list[str] | None = None, file_type: str = "parquet"):
        """Load a dataframe from a file.

        Parameters
        ----------
        file_path : str
            The path to the file. The path is relative to the `data_dir` that was passed to the constructor.
        file_type : str, optional
            The file type. By default `parquet`.

        Returns
        -------
        df : pd.DataFrame
            The loaded dataframe.

        Raises
        ------
        FileNotFoundError
            If the file type is not supported it will raise a error instead of returning a empty dataframe.
        """
        path = self.data_dir / file_path

        if not path.exists():
            logger.error(f"The file {path} does not exist.")
            raise FileNotFoundError(f"The file {path} does not exist.")

        if file_type == "parquet":
            return pd.read_parquet(path, columns=columns)

        if file_type == "feather":
            return pd.read_feather(path, columns=columns)

        if file_type == "csv":
            if columns is not None:
                logger.warning("The columns parameter is ignored when loading a csv file.")
            return pd.read_csv(path)

        raise ValueError(f"The file type {file_type} is not supported (yet).")

    def check_file_exists(self, file_path: str):
        """Check if a file exists.

        Parameters
        ----------
        file_path : str
            The path to the file. The path is relative to the `data_dir` that was passed to the constructor.

        Returns
        -------
        exists : bool
            True if the file exists, False otherwise.
        """
        path = self.data_dir / file_path
        return path.exists()

    def _create_directory(self, dir_path: str | Path):
        """Create a directory.

        Parameters
        ----------
        dir_path : str
            The path to the directory. The path is relative to the `data_dir` that was passed to the constructor.
        """
        path = self.data_dir / dir_path
        path.mkdir(parents=True, exist_ok=True)

    def save_dataframe(self, df: pd.DataFrame, file_path: str, file_type: str = "parquet"):
        """Save a dataframe to a file.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to save.
        file_path : str
            The path to the file. The path is relative to the `data_dir` that was passed to the constructor.
        file_type : str, optional
            The file type. By default `parquet`.

        Raises
        ------
        ValueError
            If the file type is not supported it will raise a error.
        """
        path = self.data_dir / file_path

        self._create_directory(path.parent)

        if file_type == "parquet":
            df.to_parquet(path)
            return

        if file_type == "feather":
            df.to_feather(path)
            return

        if file_type == "csv":
            df.to_csv(path)
            return

        raise ValueError(f"The file type {file_type} is not supported (yet).")
