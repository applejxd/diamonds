import inspect
import os
from logging import Logger, getLogger, DEBUG, Formatter, StreamHandler, FileHandler
from datetime import datetime


class SelfLogger:
    """
    シングルトンパターン
    """
    # シングルトンパターン
    _unique_instance = None
    formatter = Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
    main_logger = None

    def __new__(cls):
        # シングルトンパターン
        raise NotImplementedError("Cannot initialize via Constructor")

    @classmethod
    def __internal_new__(cls):
        """
        ユーザ定義の内部専用 new
        :return: インスタンス
        """
        # オーバーライド前の new を呼ぶ
        return super().__new__(cls)

    @classmethod
    def get_inst(cls):
        # シングルトンパターン
        if not cls._unique_instance:
            cls._unique_instance = cls.__internal_new__()
        return cls._unique_instance

    @classmethod
    def _get_file_handler(cls, name: str):
        log_file_name = name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = FileHandler(f"{os.path.dirname(__file__)}/../logs/{log_file_name}.log")
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(cls.formatter)
        return file_handler

    @classmethod
    def _get_stream_handler(cls):
        stream_handler = StreamHandler()
        stream_handler.setLevel(DEBUG)
        stream_handler.setFormatter(cls.formatter)
        return stream_handler

    @classmethod
    def _get_main_logger(cls, name: str) -> Logger:
        logger_ins = getLogger(name)
        logger_ins.setLevel(DEBUG)
        logger_ins.propagate = False

        file_handler = cls._get_file_handler(name)
        logger_ins.addHandler(file_handler)
        stream_handler = cls._get_stream_handler()
        logger_ins.addHandler(stream_handler)

        return logger_ins

    @classmethod
    def get_logger(cls, name: str) -> Logger:
        if cls.main_logger is None:
            cls.main_logger = cls._get_main_logger(name)
            return cls.main_logger
        else:
            return cls.main_logger.getChild(name)
