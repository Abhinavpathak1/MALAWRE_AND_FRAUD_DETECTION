"""
inference.py — Loads trained models and runs full hybrid analysis.
Used by app.py for live scanning.
"""

import os
import json
import joblib
import numpy as np
from features import (
    FeatureExtractor, analyze_urls,
    URGENCY_WORDS, FINANCIAL_WORDS, THREAT_WORDS,
    SOCIAL_ENG_WORDS, IMPERSONATION_BRANDS, MANIPULATION_PHRASES
)

MODELS_DIR = "models"


def load_models():
    """Load all trained models from disk."""
    required = [
        "m1_tfidf_lr.pkl", "m2_chargram_svc.pkl",
        "m3_random_forest.pkl", "m4_gradient_boost.pkl",
        "m5_mlp_neural_net.pkl", "m6_naive_bayes.pkl",
        "feature_extractor.pkl", "scaler.pkl", "metadata.json",
    ]
    missing = [f for f in required if not os.path.exists(f"{MODELS_DIR}/{f}")]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}\n"
            "Please run:  python train_model.py  first."
        )

    with open(f"{MODELS_DIR}/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "m1":  joblib.load(f"{MODELS_DIR}/m1_tfidf_lr.pkl"),
        "m2":  joblib.load(f"{MODELS_DIR}/m2_chargram_svc.pkl"),
        "m3":  joblib.load(f"{MODELS_DIR}/m3_random_forest.pkl"),
        "m4":  joblib.load(f"{MODELS_DIR}/m4_gradient_boost.pkl"),
        "m5":  joblib.load(f"{MODELS_DIR}/m5_mlp_neural_net.pkl"),
        "m6":  joblib.load(f"{MODELS_DIR}/m6_naive_bayes.pkl"),
        "fe":  joblib.load(f"{MODELS_DIR}/feature_extractor.pkl"),
        "sc":  joblib.load(f"{MODELS_DIR}/scaler.pkl"),
        "meta": meta,
    }


# ──────────────────────────────────────────────────────────────
# RULE-BASED NLP (runs in parallel with ML models)
# ──────────────────────────────────────────────────────────────

def rule_based_scores(text: str) -> dict:
    t = text.lower()
    words = text.split()
    n = max(len(words), 1)

    urgency_hits  = [kw for kw in URGENCY_WORDS if kw in t]
    fin_hits      = [kw for kw in FINANCIAL_WORDS if kw in t]
    threat_hits   = [kw for kw in THREAT_WORDS if kw in t]
    social_hits   = [kw for kw in SOCIAL_ENG_WORDS if kw in t]
    imp_hits      = [b  for b  in IMPERSONATION_BRANDS if b in t]
    manip_hits    = [p  for p  in MANIPULATION_PHRASES if p in t]

    urgency_score  = min(100, len(urgency_hits)  * 16 + text.count("!") * 4)
    financial_score= min(100, len(fin_hits)      * 12)
    threat_score   = min(100, len(threat_hits)   * 18)
    social_score   = min(100, len(social_hits)   * 22)
    imp_score      = min(100, len(imp_hits)       * 14)
    manip_score    = min(100, len(manip_hits)     * 14)

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    ling_anomaly = min(100, int(
        caps_ratio * 60 +
        text.count("!") / n * 150 +
        (20 if len([w for w in words if w.isupper() and len(w) > 2]) > 2 else 0)
    ))

    manipulation = min(100, int(
        urgency_score  * 0.35 +
        threat_score   * 0.30 +
        social_score   * 0.20 +
        manip_score    * 0.15
    ))

    # Simple sentiment
    neg = ["urgent", "warning", "alert", "danger", "critical", "threat",
           "arrest", "suspend", "freeze", "close", "delete", "expire", "penalty"]
    pos = ["safe", "secure", "verified", "confirmed", "thank", "welcome",
           "approved", "delivered", "success", "scheduled"]
    neg_c = sum(1 for w in neg if w in t)
    pos_c = sum(1 for w in pos if w in t)
    sentiment = (pos_c - neg_c) / max(neg_c + pos_c, 1)
    sentiment = max(-1.0, min(1.0, sentiment))

    all_flags = urgency_hits + fin_hits[:3] + threat_hits + social_hits[:2]
    flagged_kw = list(dict.fromkeys(all_flags))[:10]

    return {
        "urgency_score":   urgency_score,
        "financial_score": financial_score,
        "threat_score":    threat_score,
        "social_score":    social_score,
        "impersonation_score": imp_score,
        "linguistic_anomaly":  ling_anomaly,
        "manipulation_score":  manipulation,
        "sentiment":       round(sentiment, 3),
        "flagged_keywords": flagged_kw,
    }


# ──────────────────────────────────────────────────────────────
# FULL ANALYSIS
# ──────────────────────────────────────────────────────────────

def analyze(text: str, models: dict) -> dict:
    """
    Run full hybrid analysis:
      • 6 ML/DL models (trained)
      • Rule-based NLP engine
      • URL feature extractor
      • Weighted ensemble voting
    """
    fe  = models["fe"]
    sc  = models["sc"]
    meta = models["meta"]
    w   = meta["ensemble_weights"]

    # Feature matrices
    X_hand = fe.transform([text])
    X_sc   = sc.transform(X_hand)

    # ── Get probabilities from each model ──
    def prob(model, X):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0][1])
        return float(model.decision_function(X)[0])

    p1 = prob(models["m1"], [text])
    p2 = prob(models["m2"], [text])
    p3 = prob(models["m3"], X_hand)
    p4 = prob(models["m4"], X_hand)
    p5 = prob(models["m5"], X_sc)
    p6 = prob(models["m6"], [text])

    # Clip to [0, 1]
    p1, p2, p3, p4, p5, p6 = [
        max(0.0, min(1.0, p)) for p in (p1, p2, p3, p4, p5, p6)
    ]

    # Rule-based probability
    rules = rule_based_scores(text)
    p_rule = min(1.0, (
        rules["urgency_score"]    * 0.22 +
        rules["financial_score"]  * 0.20 +
        rules["threat_score"]     * 0.18 +
        rules["social_score"]     * 0.15 +
        rules["impersonation_score"] * 0.15 +
        rules["linguistic_anomaly"]  * 0.10
    ) / 100)

    # URL risk
    url_details = analyze_urls(text)
    url_risk = max((u["score"] for u in url_details), default=0) / 100

    # ── Ensemble: weighted average ──
    ensemble = (
        p1 * w["m1"] +
        p2 * w["m2"] +
        p3 * w["m3"] +
        p4 * w["m4"] +
        p5 * w["m5"] +
        p6 * w["m6"]
    )
    # Boost ensemble if URL is highly suspicious
    ensemble = min(1.0, ensemble + url_risk * 0.12 + p_rule * 0.08)

    fraud_prob = int(round(ensemble * 100))
    is_fraud   = ensemble >= 0.42

    # ── Risk score ──
    risk_score = min(100, int(
        ensemble * 72 +
        url_risk * 15 +
        rules["urgency_score"]   * 0.08 +
        rules["linguistic_anomaly"] * 0.05
    ))

    # ── Verdict ──
    if   risk_score >= 80: verdict = "CRITICAL DANGER"
    elif risk_score >= 60: verdict = "HIGH RISK"
    elif risk_score >= 38: verdict = "MODERATE RISK"
    elif risk_score >= 18: verdict = "LOW RISK"
    else:                  verdict = "SAFE"

    # ── Threat categories ──
    threat_cats = [
        {"name": "Phishing",             "score": min(100, int(p1*55 + url_risk*45)),   "detected": p1 > 0.45 or url_risk > 0.45},
        {"name": "Social Engineering",   "score": rules["social_score"],                "detected": rules["social_score"] > 28},
        {"name": "Financial Scam",       "score": rules["financial_score"],             "detected": rules["financial_score"] > 28},
        {"name": "Urgency Manipulation", "score": rules["urgency_score"],               "detected": rules["urgency_score"] > 22},
        {"name": "Impersonation",        "score": rules["impersonation_score"],         "detected": rules["impersonation_score"] > 18},
        {"name": "Malware / Ransomware", "score": min(100, int(url_risk*80 + p2*20)),   "detected": url_risk > 0.50},
        {"name": "Identity Theft",       "score": min(100, int(p3*50 + rules["financial_score"]*0.5)), "detected": p3 > 0.55},
        {"name": "Linguistic Anomaly",   "score": rules["linguistic_anomaly"],          "detected": rules["linguistic_anomaly"] > 28},
    ]

    # ── ML signal breakdown ──
    model_info = meta["models"]
    ml_signals = [
        {"name": model_info["m1"]["desc"], "prob": p1, "pct": int(p1*100),
         "triggered": p1 > 0.45,
         "detail": "Word & phrase pattern fraud classification (TF-IDF 5000 features, trigrams)"},
        {"name": model_info["m2"]["desc"], "prob": p2, "pct": int(p2*100),
         "triggered": p2 > 0.45,
         "detail": "Character-level (2–5 gram) pattern detection — catches obfuscated scam text"},
        {"name": model_info["m3"]["desc"], "prob": p3, "pct": int(p3*100),
         "triggered": p3 > 0.45,
         "detail": f"300-tree ensemble on {len(fe._extract(''))} handcrafted features"},
        {"name": model_info["m4"]["desc"], "prob": p4, "pct": int(p4*100),
         "triggered": p4 > 0.45,
         "detail": "Sequential boosting on handcrafted features — excels at subtle patterns"},
        {"name": model_info["m5"]["desc"], "prob": p5, "pct": int(p5*100),
         "triggered": p5 > 0.45,
         "detail": "3-layer neural network (Input→128→64→32→Output), ReLU, Adam optimizer"},
        {"name": model_info["m6"]["desc"], "prob": p6, "pct": int(p6*100),
         "triggered": p6 > 0.45,
         "detail": "Probabilistic bigram model — fast baseline using Bayes theorem"},
        {"name": "Rule-Based NLP Engine", "prob": p_rule, "pct": int(p_rule*100),
         "triggered": p_rule > 0.35,
         "detail": "Expert lexicon matching across urgency, financial, threat, and social-engineering patterns"},
    ]

    # ── Deep learning insights ──
    dl_insights = {
        "urgency_level":    rules["urgency_score"],
        "manipulation":     rules["manipulation_score"],
        "linguistic_anomaly": rules["linguistic_anomaly"],
        "financial_exposure": rules["financial_score"],
        "social_engineering": rules["social_score"],
        "sentiment":        rules["sentiment"],
        "mlp_confidence":   int(p5 * 100),
        "ensemble_certainty": int(abs(ensemble - 0.5) * 200),  # distance from decision boundary
    }

    # ── Recommendations ──
    recs = []
    if verdict in ("CRITICAL DANGER", "HIGH RISK"):
        recs.append("Do NOT click any links or call any phone numbers in this message.")
        recs.append("Report this to your IT/security team or relevant authorities immediately.")
        recs.append("Block the sender and delete this message without replying.")
    if url_risk > 0.4:
        recs.append("Avoid all URLs in this message — they show signs of malicious hosting (suspicious domain, no HTTPS, or IP-based).")
    if rules["financial_score"] > 25:
        recs.append("Never share bank details, SSN, gift card codes, or payment information via unsolicited messages.")
    if rules["impersonation_score"] > 15:
        recs.append("If this claims to be from a known brand, visit their official website directly — do not use links in this message.")
    if verdict == "MODERATE RISK":
        recs.append("Treat this message with caution. Verify the sender through an independent channel before taking any action.")
    if verdict in ("SAFE", "LOW RISK"):
        recs.append("This message appears legitimate. Always stay vigilant for unexpected requests for personal information.")
    recs = (recs or ["No specific threats detected. Continue practicing safe messaging habits."])[:5]

    # ── Explanation ──
    top_model = max(ml_signals[:6], key=lambda x: x["pct"])
    if is_fraud:
        explanation = (
            f"This content was classified as potentially FRAUDULENT by {sum(1 for s in ml_signals[:6] if s['triggered'])} "
            f"out of 6 ML/DL models, with a weighted ensemble fraud probability of {fraud_prob}%. "
            f"The strongest signal was {top_model['name']} ({top_model['pct']}%). "
            f"Key triggers: {', '.join(rules['flagged_keywords'][:4]) or 'pattern-based signals'}."
        )
    else:
        models_safe = sum(1 for s in ml_signals[:6] if not s["triggered"])
        explanation = (
            f"This content appears {verdict.lower()} — {models_safe}/6 models classified it as legitimate. "
            f"Ensemble fraud probability: {fraud_prob}%. "
            f"{'No significant threat indicators found.' if risk_score < 20 else 'Some mild signals present but below fraud threshold.'}"
        )

    return {
        "risk_score":       risk_score,
        "verdict":          verdict,
        "fraud_probability": fraud_prob,
        "is_fraud":         is_fraud,
        "explanation":      explanation,
        "flagged_keywords": rules["flagged_keywords"],
        "threat_categories": threat_cats,
        "ml_signals":       ml_signals,
        "dl_insights":      dl_insights,
        "url_details":      url_details,
        "url_risk":         int(url_risk * 100),
        "recommendations":  recs,
        "raw_probs": {
            "m1_lr":       round(p1, 4),
            "m2_svc":      round(p2, 4),
            "m3_rf":       round(p3, 4),
            "m4_gb":       round(p4, 4),
            "m5_mlp":      round(p5, 4),
            "m6_nb":       round(p6, 4),
            "rules":       round(p_rule, 4),
            "ensemble":    round(ensemble, 4),
        },
    }
