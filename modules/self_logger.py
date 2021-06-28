from logging import Logger, getLogger, DEBUG, Formatter, StreamHandler, FileHandler
from datetime import datetime


class SelfLogger:
    # 出力ファイル名
    log_file_name = ""
    formatter = Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

    @classmethod
    def get_file_handler(cls):
        if not cls.log_file_name:
            cls.log_file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_handler = FileHandler(f"../logs/{cls.log_file_name}.log")
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(cls.formatter)
        return file_handler

    @classmethod
    def get_stream_handler(cls):
        stream_handler = StreamHandler()
        stream_handler.setLevel(DEBUG)
        stream_handler.setFormatter(cls.formatter)
        return stream_handler

    @classmethod
    def make_logger(cls, name: str) -> Logger:
        logger_ins = getLogger(name)
        logger_ins.setLevel(DEBUG)
        logger_ins.propagate = False

        file_handler = cls.get_file_handler()
        logger_ins.addHandler(file_handler)
        stream_handler = cls.get_stream_handler()
        logger_ins.addHandler(stream_handler)

        return logger_ins
