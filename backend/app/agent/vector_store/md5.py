import threading
import os
from typing import BinaryIO, Optional
import hashlib

from app.config import config


class FileMD5Manager:
    """
    文件MD5计算 + MD5记录文件管理器
    封装：计算md5、查询是否存在、新增记录、删除记录
    """
    def __init__(self, md5_store_path: Optional[str] = None):
        # 初始化存储路径
        if md5_store_path is None:
            self._store_path = config.MD5_PATH
        else:
            self._store_path = md5_store_path
        # 文件读写锁，防止多线程并发读写md5.txt造成错乱
        self._file_lock = threading.Lock()

    @staticmethod
    def get_file_md5(fp: BinaryIO, chunk_size: int = 65536) -> str:
        """同步流式计算文件对象MD5"""
        md5_obj = hashlib.md5()
        while chunk := fp.read(chunk_size):
            md5_obj.update(chunk)
        return md5_obj.hexdigest()

    def exists_md5(self, md5_str: str) -> bool:
        """检测md5是否已经存在记录"""
        with self._file_lock:
            if not os.path.exists(self._store_path):
                # 不存在则创建空文件
                open(self._store_path, "w", encoding="utf-8").close()
                return False

            with open(self._store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_str:
                        return True
            return False

    def save_md5(self, md5_str: str) -> None:
        """追加写入md5记录"""
        with self._file_lock:
            with open(self._store_path, "a", encoding="utf-8") as f:
                f.write(md5_str + "\n")

    def remove_md5(self, target_md5: str) -> bool:
        """
        删除指定md5记录
        :return: True找到并删除 / False不存在
        """
        with self._file_lock:
            if not os.path.exists(self._store_path):
                return False

            keep_lines = []
            found = False
            with open(self._store_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line == target_md5:
                        found = True
                        continue
                    keep_lines.append(raw_line)

            if not found:
                return False

            with open(self._store_path, "w", encoding="utf-8") as fw:
                fw.writelines(keep_lines)
            return found
