from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
FRONT = os.path.join(BASE, '..', 'frontend')
app = Flask(__name__, static_folder=FRONT, static_url_path='')
state = {}


def make_preprocessor(X):
    cats = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    transformers = []
    if cats:
        try:
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        except TypeError:
            enc = OneHotEncoder(handle_unknown='ignore', sparse=True)
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', enc)]), cats))
    if nums:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), nums))
    # Constant encoded/numeric features are removed automatically.
    return Pipeline([
        ('columns', ColumnTransformer(transformers, remainder='drop')),
        ('feature_selection', VarianceThreshold(threshold=0.0))
    ])


def feature_names(pre):
    return list(pre.get_feature_names_out())


def extract_rules(tree, features, classes):
    """Extract human-readable rules without Python recursion.
    Iterative traversal avoids RecursionError for deep trees on large datasets.
    """
    t = tree.tree_
    out = []
    stack = [(0, [])]
    while stack:
        node, conditions = stack.pop()
        if t.feature[node] != -2:
            f = features[t.feature[node]]
            q = t.threshold[node]
            # Push right first so left branch is processed first.
            stack.append((t.children_right[node], conditions + [f'({f} > {q:.4g})']))
            stack.append((t.children_left[node], conditions + [f'({f} <= {q:.4g})']))
        else:
            counts = t.value[node][0]
            k = int(np.argmax(counts))
            total = float(counts.sum())
            confidence = float(counts[k] / total) if total else 0.0
            support = float(t.n_node_samples[node] / t.n_node_samples[0]) if t.n_node_samples[0] else 0.0
            out.append({
                'rule': ' AND '.join(conditions) or 'TRUE',
                'prediction': str(classes[k]),
                'confidence': round(confidence, 4),
                'support': round(support, 4),
                'samples': int(t.n_node_samples[node])
            })
    return sorted(out, key=lambda x: (-x['confidence'], -x['support']))

def metric_dict(y_true, pred):
    # Macro averages are intentionally used for the headline metrics.
    # Weighted recall is mathematically equal to accuracy in single-label
    # classification, which made the old UI appear to show duplicate values.
    return {
        'accuracy': round(float(accuracy_score(y_true, pred)), 4),
        'precision': round(float(precision_score(y_true, pred, average='macro', zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, pred, average='macro', zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, pred, average='macro', zero_division=0)), 4)
    }


def split_data(X, y, test_size):
    strat = y if y.value_counts().min() >= 2 else None
    return train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=42,
        stratify=strat
    )


def build_pipeline(X, algorithm='decision_tree', criterion='gini', max_depth=None, min_samples=2):
    # Decision Tree only. Parameters are adapted for dataset size to keep training
    # responsive on large datasets without unnecessarily restricting small datasets.
    n = max(1, len(X))
    adaptive_min = max(int(min_samples), 2 if n < 5000 else (3 if n < 50000 else 5))
    adaptive_depth = max_depth
    if adaptive_depth is None:
        if n >= 500000:
            adaptive_depth = 24
        elif n >= 100000:
            adaptive_depth = 22
        elif n >= 50000:
            adaptive_depth = 20
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=adaptive_depth,
        min_samples_split=adaptive_min,
        min_samples_leaf=max(1, adaptive_min // 2),
        class_weight=None,
        random_state=42,
    )
    return Pipeline([('preprocessor', make_preprocessor(X)), ('model', model)])

def recategorize_target(df, target):
    """Make high-cardinality numeric targets usable for classification.
    Low-cardinality targets are preserved exactly. Large numeric target spaces are
    quantile-binned into a dataset-size-aware number of categories.
    """
    s = df[target]
    if not pd.api.types.is_numeric_dtype(s):
        return df.copy(), {'enabled': False, 'original_classes': int(s.nunique(dropna=True)), 'classes': int(s.nunique(dropna=True))}
    unique = int(s.nunique(dropna=True))
    n = len(df)
    if unique <= max(12, int(np.sqrt(max(n, 1)))):
        return df.copy(), {'enabled': False, 'original_classes': unique, 'classes': unique}
    # Increase granularity with dataset size, capped to keep classification practical.
    bins = 3 if n < 5000 else 5 if n < 50000 else 7 if n < 250000 else 9
    out = df.copy()
    try:
        labels = [f'Range {i+1}' for i in range(bins)]
        out[target] = pd.qcut(pd.to_numeric(out[target], errors='coerce'), q=bins, labels=labels, duplicates='drop')
        info = {'enabled': True, 'original_classes': unique, 'classes': int(out[target].nunique(dropna=True)), 'bins': int(out[target].nunique(dropna=True)), 'method': 'Quantile re-categorization'}
        return out, info
    except Exception:
        return df.copy(), {'enabled': False, 'original_classes': unique, 'classes': unique}

def prepare_xy(df, target):
    # Preprocessing applied before model training.
    clean = df.drop_duplicates().copy()
    drop_cols = [target] + [c for c in state.get('id_cols', []) if c != target]
    X = clean.drop(columns=drop_cols).copy()
    y = clean[target].astype(str)

    # Outlier handling: clip numeric predictors to the IQR fences rather than
    # deleting rows, preserving the size of large datasets.
    for c in X.select_dtypes(include=np.number).columns:
        s = pd.to_numeric(X[c], errors='coerce')
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            X[c] = s.clip(lo, hi)

    # Feature selection: remove constant predictor columns.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    if X.shape[1] == 0:
        raise ValueError('No predictor columns remain after removing the target and identifier columns.')
    if y.nunique() < 2:
        raise ValueError('Target must contain at least two classes.')
    return X, y, drop_cols


def preprocessing_report(df, target, id_cols):
    work = df.copy()
    missing_by_column = {str(k): int(v) for k, v in work.isna().sum().items() if int(v) > 0}
    duplicate_rows = int(work.duplicated().sum())
    numeric_cols = work.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in work.columns if c not in numeric_cols]
    predictor_cols = [c for c in work.columns if c != target and c not in id_cols]

    # IQR outlier diagnostics and a clipping count. Clipping is used instead of
    # deleting records so a large dataset does not lose many students.
    outlier_counts = {}
    clipped_values = 0
    for c in numeric_cols:
        if c == target:
            continue
        s = pd.to_numeric(work[c], errors='coerce').dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[c] = 0
            continue
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        mask = (pd.to_numeric(work[c], errors='coerce') < lo) | (pd.to_numeric(work[c], errors='coerce') > hi)
        n = int(mask.sum())
        outlier_counts[c] = n
        clipped_values += n

    # Feature selection diagnostics: constant predictor columns are removed.
    constant_cols = [c for c in predictor_cols if work[c].nunique(dropna=True) <= 1]
    selected_predictors = [c for c in predictor_cols if c not in constant_cols]
    class_dist = work[target].astype(str).value_counts().to_dict() if target in work.columns else {}
    return {
        'missing_total': int(work.isna().sum().sum()),
        'missing_by_column': missing_by_column,
        'duplicates': duplicate_rows,
        'categorical_columns': categorical_cols,
        'numeric_columns': numeric_cols,
        'identifier_columns': id_cols,
        'outlier_counts': outlier_counts,
        'outlier_values_detected': int(clipped_values),
        'outlier_method': 'IQR clipping (1.5 × IQR)',
        'constant_columns_removed': constant_cols,
        'selected_features': selected_predictors,
        'feature_selection_method': 'Remove constant/zero-variance predictor columns',
        'class_distribution': [{'class': str(k), 'count': int(v)} for k, v in class_dist.items()]
    }

def build_probability_calibrator(pipe, X_train, y_train):
    """Build a separate calibrated probability model for user-facing predictions.

    A Decision Tree leaf can contain only one class, so the raw tree probability
    can legitimately become 100% even when the model's held-out accuracy is much
    lower.  That is useful for rule confidence, but it is too extreme to present
    as the probability of a new student's class.  Sigmoid calibration estimates
    probabilities on cross-validation folds and produces much more realistic
    class probabilities while leaving the main tree and its evaluation metrics
    unchanged.
    """
    min_class_count = int(y_train.value_counts().min()) if len(y_train) else 0
    if min_class_count < 2:
        return None
    cv_folds = min(3, min_class_count)
    try:
        calibrator = CalibratedClassifierCV(
            estimator=pipe, method='sigmoid', cv=cv_folds, n_jobs=-1
        )
    except TypeError:  # scikit-learn versions that still use base_estimator
        calibrator = CalibratedClassifierCV(
            base_estimator=pipe, method='sigmoid', cv=cv_folds, n_jobs=-1
        )
    calibrator.fit(X_train, y_train)
    return calibrator


def train_once(X, y, test_size, algorithm='decision_tree', criterion='gini', max_depth=None, min_samples=2):
    Xtr, Xte, ytr, yte = split_data(X, y, test_size)
    pipe = build_pipeline(X, algorithm, criterion, max_depth, min_samples)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    classes = list(pipe.named_steps['model'].classes_)
    metrics = metric_dict(yte, pred)
    metrics.update({
        'train_rows': int(len(ytr)),
        'test_rows': int(len(yte)),
        'classes': [str(x) for x in classes],
        'confusion_matrix': confusion_matrix(yte, pred, labels=classes).tolist()
    })
    return pipe, metrics, Xtr, Xte, ytr, yte


@app.get('/')
def home():
    return send_from_directory(FRONT, 'index.html')


@app.post('/api/upload')
def upload():
    try:
        f = request.files.get('file')
        if not f or not f.filename.lower().endswith('.csv'):
            return jsonify(error='Please upload a CSV file.'), 400
        df = pd.read_csv(f)
        original_rows = len(df)
        df = df.dropna(how='all')
        for c in df.select_dtypes(include='object').columns:
            df[c] = df[c].astype(str).str.strip()

        if len(df.columns) < 2:
            return jsonify(error='Dataset needs at least 2 columns.'), 400

        # Avoid one-hot encoding unique identifiers (e.g. Student_ID).
        id_cols = []
        id_words = (
            'id', 'student_id', 'studentid', 'roll', 'roll_no', 'rollno',
            'registration', 'reg_no', 'regno', 'admission', 'serial', 'sr_no'
        )
        for c in df.columns:
            cl = str(c).lower().replace(' ', '_')
            unique_ratio = df[c].nunique(dropna=False) / max(len(df), 1)
            if any(w in cl for w in id_words) and unique_ratio > 0.5:
                id_cols.append(c)

        state.clear()
        state['df'] = df
        state['id_cols'] = id_cols

        missing = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        numeric_count = int(len(df.select_dtypes(include=np.number).columns))
        categorical_count = int(len(df.columns) - numeric_count)

        default_target = df.columns[-1]
        prep = preprocessing_report(df, default_target, id_cols)

        return jsonify(
            filename=f.filename,
            rows=len(df),
            original_rows=original_rows,
            missing_values=missing,
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_count,
            categorical_columns=categorical_count,
            ignored_id_columns=id_cols,
            columns=[
                {'name': c, 'dtype': str(df[c].dtype), 'unique': int(df[c].nunique())}
                for c in df.columns
            ],
            preview=df.head(8).where(pd.notna(df.head(8)), None).to_dict('records'),
            preprocessing=prep
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post('/api/remove')
def remove_dataset():
    state.clear()
    return jsonify(ok=True)


@app.post('/api/train')
def train():
    try:
        if 'df' not in state:
            return jsonify(error='Upload a dataset first.'), 400
        b = request.get_json() or {}
        df = state['df']
        target = b.get('target')
        if target not in df.columns:
            return jsonify(error='Please select a valid target column.'), 400

        model_df, recat = recategorize_target(df, target)
        X, y, drop_cols = prepare_xy(model_df, target)
        test_size = float(b.get('test_size', .2))
        if not 0.1 <= test_size <= 0.5:
            return jsonify(error='Test size must be between 10% and 50%.'), 400

        algorithm = 'decision_tree'
        criterion = b.get('criterion', 'gini')
        max_depth = b.get('max_depth')
        min_samples = int(b.get('min_samples', 2))

        pipe, metrics, Xtr, Xte, ytr, yte = train_once(
            X, y, test_size, 'decision_tree', criterion, max_depth, min_samples
        )
        # Keep the Decision Tree itself as the classifier used for rules and
        # evaluation, but use a calibrated copy only for probability display.
        # This prevents pure tree leaves from showing misleading 100% confidence.
        calibrator = build_probability_calibrator(pipe, Xtr, ytr)
        model = pipe.named_steps['model']
        pre = pipe.named_steps['preprocessor']
        fn = feature_names(pre)

        rr = extract_rules(model, fn, model.classes_)
        rule_source = 'Decision Tree'

        imp = [
            {'feature': fn[i], 'importance': round(float(v), 5)}
            for i, v in enumerate(model.feature_importances_)
        ]
        imp.sort(key=lambda x: -x['importance'])

        state.update(
            target=target,
            recategorization=recat,
            model=pipe,
            probability_calibrator=calibrator,
            metrics=metrics,
            rules=rr,
            features=list(X.columns),
            classes=list(model.classes_),
            algorithm=algorithm,
            criterion=criterion,
            test_size=test_size,
            max_depth=max_depth,
            min_samples=min_samples,
            rule_source=rule_source,
            importances=imp
        )

        return jsonify(
            metrics=metrics,
            recategorization=recat,
            rules=rr,
            rule_source=rule_source,
            importances=imp[:20],
            preprocessing={
                **preprocessing_report(df, target, state.get('id_cols', [])),
                'input_rows': int(len(df)),
                'training_rows': int(len(Xtr)),
                'testing_rows': int(len(Xte)),
                'ignored_id_columns': state.get('id_cols', []),
                'predictor_columns': int(len(X.columns)),
                'encoded_feature_count': int(len(fn))
            }
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post('/api/compare')
def compare():
    """Run lightweight Decision Tree comparisons.
    Large datasets are sampled once for comparisons; the main model remains trained
    on the full dataset. This keeps the visualization tab responsive.
    """
    try:
        if 'df' not in state:
            return jsonify(error='Upload a CSV first.'), 400
        b = request.get_json() or {}
        target = b.get('target') or state.get('target')
        if target not in state['df'].columns:
            return jsonify(error='Please select a valid target column.'), 400
        compare_df, _ = recategorize_target(state['df'], target)
        X, y, _ = prepare_xy(compare_df, target)
        n_full = len(X)
        cap = 25000 if n_full > 25000 else n_full
        if n_full > cap:
            rng = np.random.default_rng(42)
            # Preserve class proportions when possible.
            try:
                from sklearn.model_selection import train_test_split as _tts
                X, _, y, _ = _tts(X, y, train_size=cap, random_state=42, stratify=y)
            except Exception:
                idx = rng.choice(n_full, cap, replace=False)
                X, y = X.iloc[idx], y.iloc[idx]

        min_samples = max(2, int(b.get('min_samples', 2)))
        current_criterion = b.get('current_criterion', 'gini')
        default_test = float(b.get('current_test_size', .2))
        requested_depth = b.get('current_depth') or None

        def compact(m):
            return {k: m[k] for k in ['accuracy','precision','recall','f1']}

        criteria = []
        for c in ['gini', 'entropy', 'log_loss']:
            _, m, *_ = train_once(X, y, default_test, 'decision_tree', c, requested_depth, min_samples)
            criteria.append({'name': c, **compact(m)})

        tests = []
        for t in [.10, .20, .30, .40]:
            _, m, *_ = train_once(X, y, t, 'decision_tree', current_criterion, requested_depth, min_samples)
            tests.append({'name': f'{int(t*100)}%', **compact(m)})

        depths = []
        depth_candidates = [2, 4, 8, 12, None]
        for d in depth_candidates:
            _, m, *_ = train_once(X, y, default_test, 'decision_tree', current_criterion, d, min_samples)
            depths.append({'name': 'Unlimited' if d is None else str(d), **compact(m)})

        dist = y.value_counts().to_dict()
        class_distribution = [{'name': str(k), 'count': int(v)} for k, v in dist.items()]
        return jsonify(criteria=criteria, test_sizes=tests, depths=depths,
                       class_distribution=class_distribution,
                       comparison_rows=int(len(X)), full_rows=int(n_full), sampled=bool(n_full > cap))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.post('/api/visual-data')
def visual_data():
    try:
        if 'df' not in state:
            return jsonify(error='Upload a dataset first.'), 400
        b = request.get_json() or {}
        df = state['df'].copy()
        target = b.get('target') or df.columns[-1]
        nrows = len(df)
        feature_limit = 4 if nrows < 5000 else 8 if nrows < 100000 else 12
        bin_count = 8 if nrows < 5000 else 12 if nrows < 100000 else 20
        work = df.sample(min(nrows, 200000), random_state=42) if nrows > 200000 else df
        histograms = []
        numeric = work.select_dtypes(include=np.number).columns.tolist()
        for c in numeric[:feature_limit]:
            s = pd.to_numeric(work[c], errors='coerce').dropna()
            if len(s) == 0: continue
            counts, edges = np.histogram(s.to_numpy(), bins=bin_count)
            bins = []
            for i, n in enumerate(counts):
                bins.append({'label': f'{edges[i]:.2g}–{edges[i+1]:.2g}', 'count': int(n)})
            histograms.append({'name': str(c), 'min': f'{float(s.min()):.3g}', 'max': f'{float(s.max()):.3g}', 'bins': bins})

        categories = []
        categorical_limit = 4 if nrows < 5000 else 6 if nrows < 100000 else 10
        for c in [x for x in work.columns if x not in numeric][:categorical_limit]:
            vc = work[c].astype(str).fillna('Missing').value_counts().head(10)
            categories.append({'name': str(c), 'values': [{'name': str(k), 'count': int(v)} for k,v in vc.items()]})

        # Only dataset-driven histograms and categorical bar charts are returned.
        # Diagnostics such as missing values/outliers remain in preprocessing, not visualization.
        return jsonify(histograms=histograms, categories=categories)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.post('/api/predict')
def predict():
    try:
        if 'model' not in state:
            return jsonify(error='Train a model first.'), 400
        row = request.get_json().get('row', {})
        one = pd.DataFrame([{c: row.get(c, '') for c in state['features']}])
        # Convert numeric fields back to numbers where possible.
        for c in one.columns:
            original = state['df'][c]
            if pd.api.types.is_numeric_dtype(original):
                one[c] = pd.to_numeric(one[c], errors='coerce')
        p = state['model'].predict(one)[0]
        out = {
            'prediction': str(p),
            'model_accuracy': float(state.get('metrics', {}).get('accuracy', 0.0))
        }
        # Use calibrated probabilities for the new-record UI. Raw Decision Tree
        # probabilities are leaf frequencies and can be exactly 0%/100% for pure
        # leaves; calibrated probabilities are a better estimate of uncertainty.
        probability_model = state.get('probability_calibrator')
        if probability_model is not None and hasattr(probability_model, 'predict_proba'):
            try:
                probs = probability_model.predict_proba(one)[0]
                prob_classes = probability_model.classes_
                ranked = sorted(zip(prob_classes, probs), key=lambda x: -x[1])
                out['probabilities'] = [
                    {'class': str(c), 'probability': round(float(v), 4),
                     'percentage': round(float(v) * 100, 2)}
                    for c, v in ranked
                ]
                out['confidence'] = round(float(ranked[0][1]) * 100, 2)
                out['granularity'] = len(ranked)
                out['probability_type'] = 'calibrated_sigmoid'
                out['recategorization'] = state.get('recategorization', {'enabled': False})
            except Exception:
                pass
        elif hasattr(state['model'], 'predict_proba'):
            try:
                probs = state['model'].predict_proba(one)[0]
                ranked = sorted(zip(state['model'].classes_, probs), key=lambda x: -x[1])
                out['probabilities'] = [
                    {'class': str(c), 'probability': round(float(v), 4),
                     'percentage': round(float(v) * 100, 2)}
                    for c, v in ranked
                ]
                out['confidence'] = round(float(ranked[0][1]) * 100, 2)
                out['granularity'] = len(ranked)
                out['probability_type'] = 'raw_tree_fallback'
                out['recategorization'] = state.get('recategorization', {'enabled': False})
            except Exception:
                pass
        return jsonify(out)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
