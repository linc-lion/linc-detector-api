import logging


class LoggerFactory:
    @staticmethod
    def create_logger(service_name, logger_name):
        return logging.getLogger("bentoml.{0}".format(logger_name))