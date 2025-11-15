import json
from pathlib import Path

FILE_PATH = Path('DATA/music_dirty_missing_vals.txt')

def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f'File not found: {FILE_PATH}')
    text = FILE_PATH.read_text(encoding='utf-8')
    if not text.strip():
        raise ValueError('File is empty; cannot transform genre column.')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f'Failed to parse JSON: {e}')

    if 'genre' not in data:
        raise KeyError("Key 'genre' not found in JSON data.")

    genre_dict = data.pop('genre')
    # Build binary rock mapping (1 if value == 'Rock', else 0; None stays 0)
    rock_dict = {}
    for k, v in genre_dict.items():
        rock_dict[k] = 1 if v == 'Rock' else 0

    data['rock'] = rock_dict

    # Write back
    FILE_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print('Replaced genre with binary rock column.')

if __name__ == '__main__':
    main()
