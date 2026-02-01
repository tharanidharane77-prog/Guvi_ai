import os

files = ['sample.mp3', 'test2.mp3', 'test3.mp3', 'test4.mp3']

for f in files:
    if os.path.exists(f):
        with open(f, 'rb') as fd:
            header = fd.read(10)
            print(f"{f}: {header.hex(' ')}")
    else:
        print(f"{f}: Not found")
