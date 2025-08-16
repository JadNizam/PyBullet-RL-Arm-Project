import csv, os

class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["episode", "return", "length"])
        self.f.flush()

    def log(self, episode, ret, length):
        self.w.writerow([episode, ret, length])
        self.f.flush()

    def close(self):
        self.f.close()
