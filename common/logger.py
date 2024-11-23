import datetime as dt
import logging
from rich.logging import RichHandler


def get_today_str(date):
    return date.strftime("%Y%m%d")


def make_logg(path, name, level, date):
    # Log format
    formatter = logging.Formatter(
        "%(asctime)s %(funcName)s Line %(lineno)d [%(levelname)s]: %(message)s"
    )

    # Log path(File)
    logpath = f"{path}{get_today_str(date)}_{name}.log"
    logger = logging.getLogger(name)

    # Check handler exists to avoid duplicates
    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(level)

    # RichHandler for console output
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(level)
    rich_handler.setFormatter(formatter)

    # FileHandler for log file output
    file_handler = logging.FileHandler(filename=logpath, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)

    return logger


class Logger:
    def __init__(self, path="./logs/", name="Preprocessing", level=logging.DEBUG, date=None):
        if date is None:
            date = dt.datetime.now()
        self.__logger = make_logg(path, name, level, date)
        self.__path = path
        self.__name = name
        self.__level = level
        self.__date = date

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return self.__name

    @property
    def level(self):
        return self.__level

    @property
    def logger(self):
        return self.__logger

    @property
    def date(self):
        return self.__date