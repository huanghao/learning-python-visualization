import os
import sys
from collections import defaultdict


files = defaultdict(list)
for fn in os.listdir('.'):
    if not fn.endswith('.jpg'):
        continue
    base = fn.split('.', 1)[0]
    files[base].append([os.stat(fn).st_size, fn])

for basename, names in files.items():
    names.sort(reverse=1)
    fn = names[0][1]
    print(fn)
