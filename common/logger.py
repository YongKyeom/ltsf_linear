import datetime as dt
import logging
from rich.logging import RichHandler


def get_today_str(date):
    import datetime as dt

    return date.strftime("%Y%m%d")


def make_logg(path, name, level, date):

    ## Log format
    formatter = "%(asctime)s %(funcName)s Line %(lineno)d [%(levelname)s]: %(message)s"
    ## Rich handler
    logging.basicConfig(format=formatter, handlers=[RichHandler(rich_tracebacks=True)])

    ## Log path(File)
    logpath = path + get_today_str(date) + "_" + name + ".log"
    logger = logging.getLogger(name)

    # Check handler exists
    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(level)

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=logpath, mode="a", encoding="utf-8")

    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


class Logger:
    def __init__(self, path="./logs/", name="Preprocessing", level=logging.DEBUG, date=None):
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
