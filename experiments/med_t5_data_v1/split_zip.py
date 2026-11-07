"""Seek over existing byte-split ZIP volumes without a concatenated copy."""
import bisect
import io
from pathlib import Path


class SplitFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = list(map(Path, paths))
        self.ends, self.files, self.position = [], {}, 0
        total = 0
        for path in self.paths:
            size = path.stat().st_size
            if size <= 0:
                raise ValueError('Empty archive volume')
            total += size
            self.ends.append(total)
        if not self.paths:
            raise ValueError('No archive volumes')

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.ends[-1] + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.ends[-1] - self.position)
        size = remaining if size < 0 else min(size, remaining)
        chunks = []
        while size > 0:
            index = bisect.bisect_right(self.ends, self.position)
            start = self.ends[index-1] if index else 0
            if index not in self.files:
                self.files[index] = self.paths[index].open('rb')
            stream = self.files[index]
            stream.seek(self.position - start)
            data = stream.read(min(size, self.ends[index] - self.position))
            if not data:
                raise EOFError('Truncated archive volume')
            chunks.append(data)
            self.position += len(data)
            size -= len(data)
        return b''.join(chunks)

    def close(self):
        for stream in self.files.values():
            stream.close()
        super().close()


def volumes(directory, prefix, count):
    paths = sorted(Path(directory).glob(prefix + '.[0-9][0-9][0-9]'))
    if [int(p.suffix[1:]) for p in paths] != list(range(1, count+1)):
        raise ValueError('Archive volume sequence differs')
    return paths
