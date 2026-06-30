
import csv, os

class CSVLogger:
    def __init__(self, path):
        self.path = path; os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','train_loss','test_acc','num_clean','num_noisy','num_complex'])
    def log(self, row):
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
