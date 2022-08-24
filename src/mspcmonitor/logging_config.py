import sys
import logging


def get_logger(name=__name__):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler("MSPCMonitor.log")

    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
