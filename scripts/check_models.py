import os, pickle

def load_and_report(dir_path, prefix):
    print('---', dir_path)
    feats_path = os.path.join(dir_path, 'nb1_features.pkl')
    if os.path.exists(feats_path):
        with open(feats_path,'rb') as f:
            feats = pickle.load(f)
        print('features count:', len(feats))
        print('first features:', feats[:10])
    else:
        print('features file missing:', feats_path)
    model_name = f'lightgbm_{prefix}_classification.pkl'
    model_path = os.path.join(dir_path, model_name)
    print('model path:', model_path, 'exists:', os.path.exists(model_path))
    if os.path.exists(model_path):
        try:
            with open(model_path,'rb') as f:
                model = pickle.load(f)
            print('model type:', type(model))
            if hasattr(model, 'predict_proba'):
                print('model supports predict_proba')
            if hasattr(model, 'feature_importances_'):
                print('has feature_importances_ (len=', len(getattr(model,'feature_importances_', [])), ')')
        except Exception as e:
            print('error loading model:', e)

if __name__ == '__main__':
    base = os.getcwd()
    det_dir = os.path.join(base, 'saved_models for cancer dectection')
    stage_dir = os.path.join(base, 'saved_models for stage classification')
    load_and_report(det_dir, 'detection')
    load_and_report(stage_dir, 'stage')
