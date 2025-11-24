#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Honeypot Detection - Prediction with XGBoost Feature Contributions
- XGBoost 모델의 내장 pred_contribs로 피처 기여도 계산
- 토큰 1개만으로도 작동
- SHAP 불필요
"""

import sys
import os

# Windows 인코딩 문제 해결
if sys.platform == 'win32':
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

import argparse
import json
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# ==================== CLI ====================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Predict honeypot risk with XGBoost feature contributions"
    )
    ap.add_argument("--model", default="honeypot_model_v12_shap.json",
                    help="학습된 XGBoost 모델 경로")
    ap.add_argument("--metadata", default="metadata_v12_shap.csv",
                    help="메타데이터 CSV 경로 (임계값 포함)")
    ap.add_argument("--input-csv", default="features_honeypot_v7.csv",
                    help="예측할 토큰 피처 CSV")
    ap.add_argument("--output-csv", default="honeypot_predictions.csv",
                    help="예측 결과 CSV 저장 경로")
    ap.add_argument("--output-json", default="honeypot_predictions_detail.json",
                    help="상세 결과 JSON 저장 경로")
    ap.add_argument("--token-addr", default=None,
                    help="특정 토큰 주소만 예측")
    ap.add_argument("--token-addr-idx", type=int, default=None,
                    help="특정 token_addr_idx만 예측")
    ap.add_argument("--top-k-features", type=int, default=5,
                    help="상위 기여 피처 개수 (기본값: 5)")
    return ap.parse_args()


# ==================== Feature Engineering ====================
def create_advanced_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("[Feature Engineering] Creating advanced features...")
    
    df = df.copy()
    df['sell_vol_per_cnt'] = df['total_sell_vol'] / (df['total_sell_cnt'] + 1)
    df['buy_vol_per_cnt'] = df['total_buy_vol'] / (df['total_buy_cnt'] + 1)
    df['sell_buy_cnt_ratio'] = df['total_sell_cnt'] / (df['total_buy_cnt'] + 1)
    df['sell_buy_vol_ratio'] = df['total_sell_vol'] / (df['total_buy_vol'] + 1)
    df['owner_sell_ratio'] = df['total_owner_sell_cnt'] / (df['total_sell_cnt'] + 1)
    df['non_owner_sell_ratio'] = df['total_non_owner_sell_cnt'] / (df['total_sell_cnt'] + 1)
    df['seller_buyer_ratio'] = df['unique_sellers'] / (df['unique_buyers'] + 1)
    df['avg_sell_per_seller'] = df['total_sell_cnt'] / (df['unique_sellers'] + 1)
    df['avg_buy_per_buyer'] = df['total_buy_cnt'] / (df['unique_buyers'] + 1)
    df['trade_balance'] = (df['total_buy_cnt'] - df['total_sell_cnt']) / (df['total_buy_cnt'] + df['total_sell_cnt'] + 1)
    df['liquidity_ratio'] = df['windows_with_activity'] / (df['total_windows'] + 1)
    df['sell_concentration'] = df['max_sell_share'] * df['total_sell_cnt']
    df['activity_intensity'] = (df['windows_with_activity'] / (df['total_windows'] + 1)) * (df['total_sell_cnt'] + df['total_buy_cnt'])
    df['vol_log_diff'] = df['total_sell_vol_log'] - df['total_buy_vol_log']
    df['block_window_ratio'] = df['total_sell_block_windows'] / (df['consecutive_sell_block_windows'] + 1)
    
    for col in ['sell_vol_per_cnt', 'buy_vol_per_cnt', 'sell_concentration']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    if verbose:
        print(f"  [OK] Added 18 engineered features")
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print("  [Cleaning] Cleaning data...")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    float32_min = np.finfo(np.float32).min
    float32_max = np.finfo(np.float32).max
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].clip(lower=float32_min, upper=float32_max)
    if verbose:
        print("  [OK] Cleaning complete.")
    return df


BASE_REMOVE = ['whale_total_pct', 'small_holders_pct', 'holder_balance_std',
                'holder_balance_cv', 'hhi_index', 'whale_count']


def parse_removed_features(x) -> list:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s in ("", "[]"):
        return []
    try:
        return list(ast.literal_eval(s))
    except:
        try:
            return json.loads(s)
        except:
            return []


def parse_threshold_levels(x) -> dict:
    if pd.isna(x):
        return {}
    s = str(x).strip()
    try:
        parsed = eval(s)
        return {k: float(v) for k, v in parsed.items()}
    except:
        return {}


def determine_risk_level(prob: float, thresholds: dict) -> str:
    if not thresholds:
        if prob >= 0.9:
            return 'CRITICAL'
        elif prob >= 0.7:
            return 'HIGH'
        elif prob >= 0.5:
            return 'MEDIUM'
        elif prob >= 0.3:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    thr_critical = thresholds.get('CRITICAL', 0.999)
    thr_high = thresholds.get('HIGH', 0.997)
    thr_medium = thresholds.get('MEDIUM', 0.6)
    thr_low = thresholds.get('LOW', 0.3)
    
    if prob >= thr_critical:
        return 'CRITICAL'
    elif prob >= thr_high:
        return 'HIGH'
    elif prob >= thr_medium:
        return 'MEDIUM'
    elif prob >= thr_low:
        return 'LOW'
    else:
        return 'VERY_LOW'


# ==================== XGBoost Feature Contributions ====================
def compute_feature_contributions(model, X, feature_names, verbose=True):
    """
    XGBoost 내장 기능으로 각 샘플의 피처 기여도 계산
    
    Returns:
        list of dicts: 각 샘플별 피처 기여도
    """
    if verbose:
        print("\n[Feature Contribution] Computing XGBoost feature contributions...")
        print(f"  * Dataset size: {len(X):,} samples")
        print(f"  * Features: {len(feature_names)}")
    
    # XGBoost DMatrix 생성
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    
    # 기여도 계산 (pred_contribs=True)
    # 결과: shape (n_samples, n_features + 1)
    # 마지막 컬럼은 bias (base value)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    
    if verbose:
        print(f"  [OK] Computed contributions shape: {contribs.shape}")
    
    # 각 샘플별로 분석
    results = []
    for i in range(len(contribs)):
        sample_contribs = contribs[i]
        
        # 마지막 값은 bias (base value)
        bias = sample_contribs[-1]
        feature_contribs = sample_contribs[:-1]
        
        # 피처별 기여도 정리
        contrib_list = []
        for j, (fname, contrib) in enumerate(zip(feature_names, feature_contribs)):
            contrib_list.append({
                'feature': fname,
                'contribution': float(contrib),
                'abs_contribution': float(abs(contrib)),
                'direction': 'increases_risk' if contrib > 0 else 'decreases_risk',
                'feature_value': float(X[i, j])
            })
        
        # 절대값 기준 정렬
        contrib_list.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        results.append({
            'bias': float(bias),
            'contributions': contrib_list
        })
    
    if verbose:
        print(f"  [OK] Analyzed {len(results)} samples")
    
    return results


def get_top_k_contributions(contributions_result, top_k=5):
    """
    각 샘플의 상위 K개 기여 피처 추출
    """
    top_features_per_sample = []
    
    for sample_result in contributions_result:
        top_k_contribs = sample_result['contributions'][:top_k]
        top_features_per_sample.append(top_k_contribs)
    
    return top_features_per_sample


# ==================== Main ====================
def main():
    args = parse_args()
    start_time = time.time()
    
    print("=" * 70)
    print("[Honeypot Detection] XGBoost Feature Contribution Analysis")
    print("=" * 70)

    # 모델 로드
    print("\n[Loading] Loading trained model...")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    print(f"  [OK] Loaded model: {model_path.name}")

    # 메타데이터 로드
    meta_path = Path(args.metadata)
    if meta_path.exists():
        meta_row = pd.read_csv(meta_path).iloc[0]
        threshold = float(meta_row.get('threshold', 0.5))
        removed_features = parse_removed_features(meta_row.get('removed_features', ""))
        threshold_levels = parse_threshold_levels(meta_row.get('threshold_dynamic_levels', "{}"))
        print(f"  [OK] Loaded metadata:")
        print(f"     * Threshold (MEDIUM): {threshold:.4f}")
        if threshold_levels:
            for level, thr in threshold_levels.items():
                print(f"     * {level}: {thr:.4f}")
    else:
        threshold = 0.5
        removed_features = []
        threshold_levels = {}
        print("  [WARNING] Metadata not found. Using defaults.")

    # 입력 데이터 로드
    print("\n[Loading] Loading data...")
    in_path = Path(args.input_csv)
    if not in_path.exists():
        print(f"[ERROR] Data file not found: {in_path}")
        return
    
    df_raw = pd.read_csv(in_path)
    print(f"  [OK] Loaded {len(df_raw):,} rows")
    
    has_token_addr_idx = 'token_addr_idx' in df_raw.columns
    has_token_addr = 'token_addr' in df_raw.columns

    # 필터링
    if args.token_addr is not None:
        if not has_token_addr:
            print("[ERROR] --token-addr requires 'token_addr' column")
            return
        df_raw = df_raw[df_raw['token_addr'].str.lower() == args.token_addr.lower()]
        print(f"  [Filter] Analyzing token: {args.token_addr}")
        if df_raw.empty:
            print("[ERROR] Token address not found")
            return

    if args.token_addr_idx is not None:
        if not has_token_addr_idx:
            print("[ERROR] --token-addr-idx requires 'token_addr_idx' column")
            return
        df_raw = df_raw[df_raw['token_addr_idx'] == args.token_addr_idx]
        print(f"  [Filter] Analyzing token_addr_idx: {args.token_addr_idx}")
        if df_raw.empty:
            print("[ERROR] Token index not found")
            return

    # 피처 처리
    df = df_raw.copy()
    keep_id_cols = []
    for c in ['token_addr_idx', 'token_addr']:
        if c in df.columns:
            keep_id_cols.append(c)
    drop_cols = keep_id_cols + (['label'] if 'label' in df.columns else [])
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    df = df[[c for c in df.columns if c not in BASE_REMOVE]]
    
    print("\n[Feature Engineering] Processing features...")
    df = create_advanced_features(df, verbose=True)
    df = clean_data(df, verbose=True)
    
    if removed_features:
        df = df[[c for c in df.columns if c not in removed_features]]
    
    if hasattr(model, "feature_names_in_"):
        need = list(model.feature_names_in_)
        for c in need:
            if c not in df.columns:
                df[c] = 0.0
        df = df[need]
    
    print(f"  [OK] Processed features: {df.shape[1]} features")

    # 예측
    print("\n[Prediction] Making predictions...")
    X = df.values
    feature_names = df.columns.tolist()
    
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    risk_levels = [determine_risk_level(p, threshold_levels) for p in proba]
    
    n_honeypots = int((pred == 1).sum())
    n_safe = int((pred == 0).sum())
    hp_rate = n_honeypots / len(pred) * 100.0
    
    print("\n  [OK] Prediction complete!")
    print(f"    [ALERT] Honeypots detected: {n_honeypots} ({hp_rate:.1f}%)")
    print(f"    [SAFE] Safe tokens: {n_safe} ({100.0 - hp_rate:.1f}%)")

    # 피처 기여도 계산
    contributions_result = compute_feature_contributions(
        model, X, feature_names, verbose=True
    )
    top_features_per_token = get_top_k_contributions(
        contributions_result, top_k=args.top_k_features
    )

    # 검증 지표
    has_labels = 'label' in df_raw.columns
    if has_labels:
        y_true = df_raw['label'].values
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        print("\n  [Validation] Metrics:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-score:  {f1:.4f}")

    # 결과 저장
    print("\n[Saving] Saving results...")
    out = pd.DataFrame({
        'token_addr_idx': df_raw['token_addr_idx'] if has_token_addr_idx else range(len(df_raw)),
        'token_addr': df_raw['token_addr'] if has_token_addr else None,
        'honeypot_probability': proba,
        'prediction': pred,
        'prediction_label': np.where(pred == 1, "HONEYPOT", "SAFE"),
        'risk_level': risk_levels,
        'threshold': threshold,
    })
    
    # 피처 기여도 추가
    for k in range(1, args.top_k_features + 1):
        out[f'top_feature_{k}'] = [
            contrib[k-1]['feature'] if len(contrib) >= k else None 
            for contrib in top_features_per_token
        ]
        out[f'top_feature_{k}_value'] = [
            contrib[k-1]['feature_value'] if len(contrib) >= k else None 
            for contrib in top_features_per_token
        ]
        out[f'top_feature_{k}_contribution'] = [
            contrib[k-1]['contribution'] if len(contrib) >= k else None 
            for contrib in top_features_per_token
        ]
        out[f'top_feature_{k}_direction'] = [
            contrib[k-1]['direction'] if len(contrib) >= k else None 
            for contrib in top_features_per_token
        ]
    
    if has_labels:
        out['actual_label'] = df_raw['label'].values
        out['correct'] = (out['prediction'] == out['actual_label']).astype(bool)
    
    out = out.sort_values('honeypot_probability', ascending=False)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"  [OK] Saved: {args.output_csv}")

    # JSON 저장
    from collections import Counter
    risk_counts = Counter(risk_levels)
    
    top10 = out.head(10).copy()
    detailed = {
        'summary': {
            'total_tokens': int(len(out)),
            'honeypots_detected': n_honeypots,
            'safe_tokens': n_safe,
            'honeypot_rate': float(hp_rate),
            'threshold_used': float(threshold),
            'threshold_levels': threshold_levels,
            'explanation_method': 'xgboost_feature_contributions',
            'risk_distribution': {
                level: int(risk_counts.get(level, 0)) 
                for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW']
            }
        },
        'top_10_risky_tokens': []
    }
    
    if has_labels:
        detailed['summary']['validation_metrics'] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
    
    for idx, (i, row) in enumerate(top10.iterrows()):
        token_detail = {
            'rank': idx + 1,
            'token_addr_idx': int(row['token_addr_idx']),
            'token_addr': (None if pd.isna(row['token_addr']) else str(row['token_addr'])),
            'probability': float(row['honeypot_probability']),
            'prediction': str(row['prediction_label']),
            'risk_level': str(row['risk_level']),
            'top_contributing_features': []
        }
        
        # 상위 기여 피처 추가
        for k in range(1, args.top_k_features + 1):
            feat = row.get(f'top_feature_{k}')
            if feat is not None and not pd.isna(feat):
                token_detail['top_contributing_features'].append({
                    'feature': str(feat),
                    'feature_value': float(row[f'top_feature_{k}_value']),
                    'contribution': float(row[f'top_feature_{k}_contribution']),
                    'direction': str(row[f'top_feature_{k}_direction'])
                })
        
        detailed['top_10_risky_tokens'].append(token_detail)
    
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved: {args.output_json}")

    # 요약 출력
    print("\n" + "=" * 70)
    print("[Complete] Analysis Complete!")
    print("=" * 70)
    print(f"\n[Summary]")
    print(f"  * Total tokens: {len(out):,}")
    print(f"  * Honeypots: {n_honeypots:,} ({hp_rate:.1f}%)")
    print(f"  * Safe: {n_safe:,} ({100.0-hp_rate:.1f}%)")
    print(f"  * Explanation: XGBoost Feature Contributions")
    
    if has_labels:
        print(f"\n[Validation]")
        print(f"  * Accuracy: {acc:.2%}")
        print(f"  * F1-score: {f1:.2%}")
    
    # 상세 출력 (최대 5개)
    print(f"\n[Detailed Analysis - Top {min(5, len(out))} Tokens]")
    print("=" * 70)
    
    for idx in range(min(5, len(out))):
        row_data = out.iloc[idx]
        token_id = row_data.get('token_addr', row_data['token_addr_idx'])
        
        print(f"\n#{idx+1}. Token: {token_id}")
        print(f"   Honeypot Probability: {row_data['honeypot_probability']:.4f} ({row_data['honeypot_probability']*100:.2f}%)")
        print(f"   Prediction: {row_data['prediction_label']}")
        print(f"   Risk Level: {row_data['risk_level']}")
        
        # 모델의 bias 값
        bias = contributions_result[idx]['bias']
        print(f"   Model Bias (Base Value): {bias:.4f}")
        
        print(f"\n   Top {args.top_k_features} Contributing Features:")
        for k in range(1, args.top_k_features + 1):
            feat = row_data.get(f'top_feature_{k}')
            if feat is not None and not pd.isna(feat):
                val = row_data[f'top_feature_{k}_value']
                contrib = row_data[f'top_feature_{k}_contribution']
                direction = row_data[f'top_feature_{k}_direction']
                
                # 기여도 시각화
                contrib_bar = '█' * int(abs(contrib) * 10) if abs(contrib) > 0.1 else '▌'
                contrib_sign = '+' if contrib > 0 else ''
                
                print(f"     {k}. {feat}")
                print(f"        Value: {val:.4f}")
                print(f"        Contribution: {contrib_sign}{contrib:.4f} {contrib_bar}")
                print(f"        Effect: {direction}")
        
        # 최종 예측 계산 설명
        total_contrib = sum([
            row_data.get(f'top_feature_{k}_contribution', 0) 
            for k in range(1, args.top_k_features + 1)
            if not pd.isna(row_data.get(f'top_feature_{k}'))
        ])
        
        print(f"\n   Prediction Calculation:")
        print(f"     Base value: {bias:.4f}")
        print(f"     + Top {args.top_k_features} contributions: {total_contrib:+.4f}")
        print(f"     + Other features: {row_data['honeypot_probability'] - bias - total_contrib:+.4f}")
        print(f"     = Final probability: {row_data['honeypot_probability']:.4f}")
        
        print("   " + "-" * 66)
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[Time] Execution time: {total_elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()