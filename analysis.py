import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter


def load_data(path):
    df = pd.read_csv(path)
    return df


def _detect_duration_event(df):
    duration_candidates = [
        'time_to_event', 'rfs_months', 'time_months', 'survival_time', 'time',
        'time_to_recurrence', 'time_to_followup', 'follow_up_time', 'followup_time'
    ]
    event_candidates = ['recurrence', 'event', 'recurrence_event', 'survival_status', 'death_event', 'status', 'event_observed']

    normalized = {str(c).strip().lower().replace(' ', '_'): c for c in df.columns}

    duration = None
    for candidate in duration_candidates:
        if candidate in normalized:
            duration = normalized[candidate]
            break

    if duration is None:
        for col in df.columns:
            lower = str(col).strip().lower().replace(' ', '_')
            if lower.startswith('time') and lower not in {'time_to_diagnosis'}:
                duration = col
                break

    if duration is None:
        numerics = [c for c in df.select_dtypes('number').columns if str(c).lower() not in {'patient_id', 'id'}]
        duration = numerics[0] if numerics else None

    event = None
    for candidate in event_candidates:
        if candidate in normalized:
            event = normalized[candidate]
            break

    if event is None and duration is not None:
        for col in df.columns:
            if str(col).lower() in {'patient_id', 'id', str(duration).lower()}:
                continue
            values = df[col].dropna()
            if values.nunique() <= 2:
                event = col
                break

    return duration, event


def _normalize_event_series(series):
    if series is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)

    values = series.astype(str).str.strip().str.lower()
    mapping = {
        'yes': 1, 'y': 1, 'true': 1, '1': 1,
        'deceased': 1, 'dead': 1, 'event': 1, 'occurred': 1,
        'no': 0, 'n': 0, 'false': 0, '0': 0,
        'survived': 0, 'alive': 0, 'censored': 0,
    }
    mapped = values.map(mapping)
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors='coerce')
        mapped = mapped.fillna(numeric)
    return mapped.fillna(0).astype(int)


def _build_cox_frame(df):
    duration_col, event_col = _detect_duration_event(df)
    if duration_col is None or event_col is None:
        return None, duration_col, event_col

    duration = pd.to_numeric(df[duration_col], errors='coerce').fillna(0)
    event = _normalize_event_series(df[event_col])
    feature_frame = df.drop(columns=[duration_col, event_col], errors='ignore')
    feature_frame = feature_frame.drop(columns=[c for c in feature_frame.columns if str(c).lower() in {'patient_id', 'id'}], errors='ignore')
    feature_frame = pd.get_dummies(feature_frame, dummy_na=False, drop_first=True)
    feature_frame = feature_frame.select_dtypes(include=['number'])

    if feature_frame.shape[1] == 0:
        return None, duration_col, event_col

    cox_df = pd.concat([
        pd.Series(duration, name=duration_col),
        pd.Series(event, name=event_col),
        feature_frame,
    ], axis=1)
    cox_df = cox_df.dropna()
    return cox_df, duration_col, event_col


def compute_key_stats(df):
    duration_col, event_col = _detect_duration_event(df)
    stats = {
        'recurrence_rate': None,
        'median_rfs_months': None,
        'rfs_12m': None,
        'rfs_24m': None,
    }

    event_series = None
    if event_col and event_col in df.columns:
        event_series = _normalize_event_series(df[event_col])
        if len(event_series) > 0:
            stats['recurrence_rate'] = float(event_series.mean())

    if duration_col in df.columns:
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(df[duration_col], event_observed=event_series if event_series is not None else None)
            stats['median_rfs_months'] = kmf.median_survival_time_
            stats['rfs_12m'] = float(kmf.predict(12))
            stats['rfs_24m'] = float(kmf.predict(24))
        except Exception:
            stats['median_rfs_months'] = None
            stats['rfs_12m'] = None
            stats['rfs_24m'] = None

    return stats


def _figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def plot_km_group_base64(df, group_col):
    duration_col, event_col = _detect_duration_event(df)
    if duration_col is None or group_col not in df.columns:
        return None

    duration = pd.to_numeric(df[duration_col], errors='coerce').fillna(0)
    event = _normalize_event_series(df[event_col]) if event_col in df.columns else None

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()
    plotted = False
    for name, group in df.groupby(group_col):
        group_idx = group.index
        try:
            group_duration = duration.loc[group_idx]
            group_event = event.loc[group_idx] if event is not None else None
            kmf.fit(group_duration, event_observed=group_event, label=str(name))
            kmf.plot_survival_function(ax=ax, ci_show=False)
            plotted = True
        except Exception:
            continue
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(f'Kaplan-Meier by {group_col.replace("_", " ").title()}')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Recurrence-free probability')
    ax.legend(title=group_col.replace('_', ' ').title(), fontsize='small')
    ax.grid(alpha=0.25)
    return _figure_to_base64(fig)


def plot_cox_forest_base64(df):
    cox_df, duration_col, event_col = _build_cox_frame(df)
    if cox_df is None or cox_df.shape[0] < 3:
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
        summary = cph.summary.reset_index()
        summary['hr'] = summary['exp(coef)']
        summary['ci_lower'] = summary['exp(coef) lower 95%']
        summary['ci_upper'] = summary['exp(coef) upper 95%']
        summary = summary.sort_values('hr')
        fig, ax = plt.subplots(figsize=(6, max(4, len(summary) * 0.4)))
        ax.errorbar(summary['hr'], range(len(summary)), xerr=[summary['hr'] - summary['ci_lower'], summary['ci_upper'] - summary['hr']], fmt='o', color='#4f46e5', ecolor='#93c5fd', capsize=4)
        ax.axvline(1, color='grey', linestyle='--', linewidth=1)
        ax.set_xscale('log')
        ax.set_xlabel('Hazard ratio (log scale)')
        ax.set_ylabel('Covariate')
        ax.grid(alpha=0.2, axis='x')
        plt.tight_layout()
        return _figure_to_base64(fig)
    except Exception:
        return None


def csv_to_html_table(path, classes='min-w-full text-sm text-left'):
    if not os.path.exists(path):
        return '<p class="text-sm text-red-600">Saved data not found.</p>'
    df = pd.read_csv(path)
    return df.to_html(index=False, classes=classes, border=0)


def ensure_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    duration_col, event_col = _detect_duration_event(df)
    # Kaplan-Meier groups
    groups = {
        'stage': 'stage',
        'treatment': None,
        'socioeconomic_status': 'socioeconomic_status',
        'follow_up_adherence': 'follow_up_adherence',
        'insurance_coverage': 'insurance_coverage',
        'smoking_status': 'smoking_status',
    }
    # If explicit treatment column missing, try chemo/radio/surgery
    if 'treatment' not in df.columns:
        if set(['chemo', 'radio', 'surgery']).intersection(df.columns):
            df['treatment'] = (
                df.get('chemo', '').astype(str) + '/' + df.get('radio', '').astype(str) + '/' + df.get('surgery', '').astype(str)
            )

    for key, col in groups.items():
        colname = col if col and col in df.columns else key if key in df.columns else None
        if colname:
            fp = os.path.join(out_dir, f'km_{key}.png')
            try:
                _plot_km_by_group(df, duration_col, event_col, colname, fp)
            except Exception:
                pass

    # Cox forest
    try:
        fp = os.path.join(out_dir, 'cox_forest.png')
        _plot_cox_forest(df, duration_col, event_col, fp)
    except Exception:
        pass


def _plot_km_by_group(df, duration_col, event_col, group_col, out_path):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for name, grouped in df.groupby(group_col):
        try:
            kmf.fit(grouped[duration_col], event_observed=grouped[event_col] if event_col in df.columns else None, label=str(name))
            kmf.plot_survival_function(ci_show=False)
        except Exception:
            continue
    plt.title(f'Kaplan-Meier by {group_col}')
    plt.xlabel('Time (months)')
    plt.ylabel('Recurrence-free probability')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_cox_forest(df, duration_col, event_col, out_path):
    # Prepare dataframe for Cox: drop non-numeric
    covariates = df.select_dtypes(include=['number']).copy()
    if duration_col in covariates.columns:
        covariates = covariates.drop(columns=[duration_col])
    if event_col in covariates.columns:
        covariates = covariates.drop(columns=[event_col])
    cox_df = df[[duration_col, event_col]].join(covariates)
    cph = CoxPHFitter()
    cph.fit(cox_df.dropna(), duration_col=duration_col, event_col=event_col)
    summary = cph.summary.reset_index()
    summary['hr'] = summary['exp(coef)']
    summary['ci_lower'] = summary['exp(coef) lower 95%']
    summary['ci_upper'] = summary['exp(coef) upper 95%']
    summary = summary.sort_values('hr')
    plt.figure(figsize=(6, max(4, len(summary)*0.4)))
    plt.errorbar(summary['hr'], range(len(summary)), xerr=[summary['hr']-summary['ci_lower'], summary['ci_upper']-summary['hr']], fmt='o')
    plt.axvline(1, color='grey', linestyle='--')
    plt.xscale('log')
    plt.xlabel('Hazard ratio (log scale)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def treatment_combo_table_html(df):
    # Build combo from chemo/radio/surgery
    cols = ['chemo', 'radio', 'surgery']
    present = [c for c in cols if c in df.columns]
    if not present:
        return '<p>No treatment combination data available.</p>'
    df['_combo'] = df[[c for c in present]].astype(str).agg('-'.join, axis=1)
    # compute 12/24/48m recurrence rates per combo using KM
    duration_col, event_col = _detect_duration_event(df)
    rows = []
    kmf = KaplanMeierFitter()
    for name, group in df.groupby('_combo'):
        try:
            kmf.fit(group[duration_col], event_observed=group[event_col] if event_col in df.columns else None)
            r12 = 1 - float(kmf.predict(12))
            r24 = 1 - float(kmf.predict(24))
            r48 = 1 - float(kmf.predict(48))
        except Exception:
            r12 = r24 = r48 = None
        rows.append({'combo': name, 'recurrence_12m': r12, 'recurrence_24m': r24, 'recurrence_48m': r48})
    table = pd.DataFrame(rows).sort_values('recurrence_24m')
    return table.to_html(index=False, classes='table table-striped')


def subgroup_summary_table_html(df):
    duration_col, event_col = _detect_duration_event(df)
    timepoints = [12,24,48]
    factors = [c for c in df.columns if df[c].nunique() < 20 and c not in [duration_col, event_col]]
    results = []
    kmf = KaplanMeierFitter()
    for factor in factors:
        for level, group in df.groupby(factor):
            row = {'factor': factor, 'level': level}
            try:
                kmf.fit(group[duration_col], event_observed=group[event_col] if event_col in df.columns else None)
                for t in timepoints:
                    row[f'recurrence_{t}m'] = 1 - float(kmf.predict(t))
            except Exception:
                for t in timepoints:
                    row[f'recurrence_{t}m'] = None
            results.append(row)
    table = pd.DataFrame(results)
    return table.to_html(index=False, classes='table table-sm')
