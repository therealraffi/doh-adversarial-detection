txt = open('detector.py').read()

# 1. Add --adversarial argument
old = '    p.add_argument("--no_nn", action="store_true", help="Skip neural network model")'
new = ('    p.add_argument("--no_nn", action="store_true", help="Skip neural network model")\n'
       '    p.add_argument("--adversarial", default=None, help="Path to adversarial flows CSV for hard negative retraining")')
txt = txt.replace(old, new)

# 2. Inject adversarial data before preprocess
old = '    X_train, X_test, y_train, y_test = preprocess('
inject = (
    '\n    # Inject adversarial hard negatives\n'
    '    if args.adversarial and os.path.exists(args.adversarial):\n'
    '        adv = pd.read_csv(args.adversarial)\n'
    '        print(f"\\n[+] Injecting {len(adv)} adversarial flows as hard negatives")\n'
    '        adv_feat = adv[[c for c in adv.columns if c in feature_cols]]\n'
    '        for col in feature_cols:\n'
    '            if col not in adv_feat.columns:\n'
    '                adv_feat[col] = 0.0\n'
    '        adv_feat = adv_feat[feature_cols]\n'
    '        adv_labels = pd.Series([1] * len(adv_feat))\n'
    '        X = pd.concat([X, adv_feat], ignore_index=True)\n'
    '        y = pd.concat([y, adv_labels], ignore_index=True)\n'
    '        print(f"[+] Dataset size after injection: {len(X)} rows")\n'
    '        print(f"[+] Class distribution: {dict(y.value_counts().sort_index())}")\n'
    '    elif args.adversarial:\n'
    '        print(f"[!] Adversarial file not found: {args.adversarial}")\n'
    '\n    X_train, X_test, y_train, y_test = preprocess('
)
txt = txt.replace(old, inject)

open('detector.py', 'w').write(txt)
print('Patched successfully.')
print('Now run:')
print('  python detector.py --l2 data/l2-total-add.csv --output ./results_adversarial --no_nn --adversarial adversarial_flows_dataset.csv')
