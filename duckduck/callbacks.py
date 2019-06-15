from glob import glob
import os
from pathlib import Path
from threading import Thread
from queue import Queue

import requests


def printJSON(objs):
    for obj in objs:
        print("Width {}, Height {}".format(obj["width"], obj["height"]))
        print("Thumbnail {}".format(obj["thumbnail"]))
        print("Url {}".format(obj["url"]))
        print("Title {}".format(obj["title"].encode('utf-8')))
        print("Image {}".format(obj["image"]))
        print("__________")


class ImageSaverCallback:
    """Puts image request in queue

    This should not be used directly, instead do:

    ```
    from pathlib import Path
    with ImageSaver(Path('./images'), overwrite=False) as callback:
        search('test', callback=callback)
    ```
    """
    def __init__(self, queue, overwritecheck):
        self.queue = queue
        self.overwritecheck = overwritecheck
        self.idx = 0

    def __call__(self, objs):
        idx = self.idx
        for obj in objs:
            while self.overwritecheck(idx):
                idx += 1
            self.queue.put((obj['image'], idx))
            idx += 1
        self.idx = idx


class ImageSaver:
    def __init__(
            self, targetdir, *,
            prefix='img',
            known_formats=('jpg', 'jpeg', 'gif', 'png', 'tiff'),
            nworkers=100,
            overwrite=False,
    ):
        self.prefix = prefix
        self.targetdir = targetdir 
        self.known_formats = known_formats
        self.threads = []
        self.queue = Queue()
        self.nworkers = nworkers
        self.overwrite = overwrite

    def _create_targetdir_if_not_exists(self):
        os.makedirs(self.targetdir, exist_ok=True)

    def _get_image(self, url, idx):
        fileformat = self._file_format(url)
        if fileformat not in self.known_formats:
            return
        try:
            r = requests.get(url, stream=True)
        except requests.exceptions.SSLError:
            print("Skipping {} (status {})".format(url, r.status_code))
            return
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print("Skipping {} (status {})".format(url, r.status_code))
            return

        path = self._save_path(idx, fileformat=fileformat)
        with open(path, 'wb') as f:
            for chunk in r:
                f.write(chunk)
        print("Saved {}".format(path))

    def _save_path(self, idx, *, fileformat='*'):
        return self.targetdir / (self.prefix + str(idx).zfill(6) + '.' + fileformat)

    def _exists_file(self, idx):
        path = self._save_path(idx)
        return any(glob(str(path)))

    def _file_format(self, url):
        baseurl = url.split('?', 1)[0]
        filename = baseurl.split('/')[-1]
        return filename.split('.')[-1].lower()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            self._get_image(*item)
            self.queue.task_done()

    def __enter__(self):
        def all_false(idx):
            False
        self._create_targetdir_if_not_exists()

        for _ in range(self.nworkers):
            t = Thread(target=self._worker)
            t.start()
            self.threads.append(t)        

        return ImageSaverCallback(
            self.queue,
            all_false if self.overwrite else self._exists_file,
        )

    def __exit__(self, *args):
        for _ in range(self.nworkers):
            self.queue.put(None)            
        print('Waiting for downloads to finish')
        for t in self.threads:
            t.join()
