import pandas as pd
import json
from tqdm import tqdm

print('Reading data...')

df = pd.read_csv('data/repos2ids_v3.4_stats.csv')

split_tokens = df['token_split'].tolist()

identifiers = []
targets = []

skipped = 0

for token in tqdm(split_tokens, desc = 'Creating examples...'):
    try:
        chars = list(token)
    except Exception as e:
        skipped += 1
        continue
    no_space_chars = []
    space_locations = []
    space = False
    for i, c in enumerate(chars):
        if chars[i] == ' ':
            space_locations.append(1)
            space = True
        else:
            no_space_chars.append(c)
            if space:
                space = False
            else:
                space_locations.append(0)
    assert len(no_space_chars) == len(space_locations)
    identifiers.append(no_space_chars)
    targets.append(space_locations)

assert len(identifiers) == len(targets)

print(f'Got {len(identifiers)} examples, skipped {skipped} examples')

print('Writing to file...')

with open('data/all_example.json', 'w+') as w:
    for identifier, target in zip(identifiers, targets):
        example = {'identifier': identifier, 'target': target}
        json.dump(example, w)
        w.write('\n')