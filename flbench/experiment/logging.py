from pathlib import Path
import csv,json
class CSVLogger:
    def __init__(self,path,fields):
        self.path=Path(path);self.fields=fields;self.path.parent.mkdir(parents=True,exist_ok=True)
        if not self.path.exists():
            with self.path.open('w',newline='') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
    def append(self,row):
        with self.path.open('a',newline='') as f:csv.DictWriter(f,fieldnames=self.fields).writerow({k:row.get(k) for k in self.fields})
def save_json(obj,path):Path(path).write_text(json.dumps(obj,indent=2,default=str))
