import os
import requests

MODELS = {
    'detection': 'REPLACE_WITH_MODEL_URL_DETECTION',
    'stage': 'REPLACE_WITH_MODEL_URL_STAGE',
    'survival': 'REPLACE_WITH_MODEL_URL_SURVIVAL',
}


def download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main():
    print('This script downloads model artifacts referenced in README.')
    for name, url in MODELS.items():
        if url.startswith('REPLACE'):
            print(f"Model '{name}' missing URL; edit {__file__} to set the URL.")
            continue
        if name == 'detection':
            dest = 'saved_models for cancer dectection/lightgbm_detection.pkl'
        elif name == 'stage':
            dest = 'saved_models for stage classification/lightgbm_stage_classification.pkl'
        else:
            dest = 'saved_models for survival analysis/nb3_cox_model.pkl'
        print(f'Downloading {name} -> {dest}')
        download(url, dest)


if __name__ == '__main__':
    main()
