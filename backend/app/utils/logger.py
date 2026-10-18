import os
import logging
from datetime import datetime

from app.config import config


# 日志格式配置 error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = 'app',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None
) -> logging.Logger:
    loggers = logging.getLogger(name)
    loggers.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if loggers.handlers:
        return loggers

    # 配置控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    loggers.addHandler(console_handler)

    # 配置文件Handler
    if not log_file:
        log_file = os.path.join(config.LOG_PATH, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    loggers.addHandler(file_handler)

    return loggers


# 快捷获取日志器
logger = get_logger()

if __name__ == '__main__':
    logger.info('信息日志')
    logger.error('错误日志')
    logger.warning('警告日志')
    logger.debug('调试日志')
