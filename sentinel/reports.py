"""实验目录、CSV 和耗时统计。"""
import csv
import math


def new_run(parent):
    parent.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name[4:]) for p in parent.iterdir()
                  if p.is_dir() and p.name.startswith('run_') and p.name[4:].isdigit()), default=0) + 1
    while True:
        path = parent / f'run_{number:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction=.95):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]
