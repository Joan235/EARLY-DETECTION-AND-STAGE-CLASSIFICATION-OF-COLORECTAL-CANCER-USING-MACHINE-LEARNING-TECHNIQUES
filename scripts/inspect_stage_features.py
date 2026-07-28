import pickle, os, json
p = os.path.join('saved_models for stage classification', 'nb1_features.pkl')
if not os.path.exists(p):
    print('MISSING', p)
else:
    try:
        data = pickle.load(open(p, 'rb'))
        print('TYPE', type(data).__name__)
        try:
            # Try to coerce to list for printing
            items = list(data)
            print(json.dumps(items, indent=2))
        except Exception:
            print(repr(data))
    except Exception as e:
        print('ERROR', e)
