"""
features.py — Feature engineering for fraud/threat detection.
Extracts handcrafted linguistic, structural, and URL features.
"""

import re
import math
import string
from collections import Counter
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ──────────────────────────────────────────────────────────────
# LEXICONS
# ──────────────────────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "expires", "limited time", "act now", "hours left",
    "deadline", "final notice", "last chance", "expire", "now or", "before it",
    "respond immediately", "within 24", "within 48", "today only", "don't wait",
    "time sensitive", "asap", "right now", "instant access", "ending soon",
    "24 hours", "48 hours", "midnight", "expire tonight", "final warning",
    "last reminder", "one last chance",
]

FINANCIAL_WORDS = [
    "bank account", "credit card", "debit card", "ssn", "social security",
    "wire transfer", "bitcoin", "crypto", "cryptocurrency", "paypal", "venmo",
    "cashapp", "zelle", "western union", "moneygram", "routing number",
    "account number", "gift card", "itunes", "google play", "amazon gift",
    "prize", "winner", "lottery", "million", "thousand dollars", "free money",
    "cash reward", "unclaimed", "grant", "inheritance", "refund", "tax refund",
    "irs", "claim your", "reward points", "investment", "returns", "profit",
    "earn money", "make money", "income", "passive income", "financial freedom",
    "no credit check", "pre-approved", "guaranteed returns",
]

THREAT_WORDS = [
    "arrested", "arrest warrant", "lawsuit", "legal action", "suspended",
    "account closed", "account locked", "account compromised", "hacked",
    "virus detected", "malware", "infected", "security breach", "blocked",
    "frozen", "terminated", "deleted", "cancelled", "legal proceedings",
    "court summons", "police", "fbi", "federal", "criminal", "investigation",
    "felony", "charges filed",
]

SOCIAL_ENG_WORDS = [
    "dear friend", "beloved", "greetings", "i am a prince", "nigerian",
    "inheritance", "deceased", "confidential", "secret agent", "only you",
    "trusted partner", "god bless", "percentage", "million usd", "million dollars",
    "help me transfer", "dying", "cancer", "widow", "diplomat", "consignment",
    "unclaimed funds", "foreign fund",
]

IMPERSONATION_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook", "netflix",
    "bank of america", "chase", "wells fargo", "citibank", "irs", "fbi", "cia",
    "social security administration", "fedex", "ups", "dhl", "usps", "customs",
    "hmrc", "royal mail", "vodafone", "o2", "hsbc", "barclays", "lloyds",
    "coinbase", "binance", "spotify", "instagram", "whatsapp",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".club", ".work",
    ".click", ".link", ".download", ".zip", ".review", ".country",
    ".science", ".vip", ".win", ".online", ".site", ".space", ".faith",
    ".stream", ".date", ".racing",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "short.link",
    "buff.ly", "dlvr.it", "ift.tt", "rebrand.ly", "bl.ink", "cutt.ly",
    "is.gd", "rb.gy", "shorturl.at",
]

TRUST_INDICATORS = [
    "https://www.", "your account", "as requested", "thank you for",
    "confirmation", "receipt", "tracking number", "order number",
    "reference number", "invoice", "statement",
]

MANIPULATION_PHRASES = [
    "act now", "don't miss", "you've been selected", "exclusively for you",
    "only you can", "this offer expires", "final opportunity", "last chance",
    "before it's too late", "while supplies last", "limited availability",
    "respond immediately", "do not ignore", "failure to respond", "or else",
]

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def extract_urls(text: str) -> list:
    return re.findall(r'https?://[^\s<>"\',;]+', text)

def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def domain_risk_score(domain: str) -> float:
    score = 0.0
    if any(tld in domain for tld in SUSPICIOUS_TLDS):
        score += 0.40
    if any(s in domain for s in URL_SHORTENERS):
        score += 0.30
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
        score += 0.40
    if domain.count('-') >= 3:
        score += 0.20
    if domain.count('.') >= 4:
        score += 0.15
    for brand in IMPERSONATION_BRANDS:
        brand_clean = brand.replace(' ', '')
        if brand_clean in domain and not domain.endswith(f".{brand_clean}.com"):
            score += 0.35
    ent = char_entropy(domain)
    if ent > 3.6:
        score += 0.20
    return min(1.0, score)

# ──────────────────────────────────────────────────────────────
# SKLEARN TRANSFORMER
# ──────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Keyword counts
    "urgency_hits", "financial_hits", "threat_hits",
    "social_eng_hits", "impersonation_hits", "manipulation_hits",
    "trust_hits",
    # Structural
    "caps_ratio", "exclamation_count", "question_count",
    "url_count", "word_count", "char_count", "avg_word_len",
    "sentence_count", "avg_sentence_len",
    # Style
    "double_space_count", "dollar_signs", "number_blobs",
    "all_caps_words", "ellipsis_count", "comma_count",
    # URL
    "has_url", "max_url_domain_risk", "avg_url_domain_risk",
    "has_http_only", "has_shortener", "has_ip_domain",
    "has_suspicious_tld", "url_entropy_avg", "longest_url_len",
    "has_brand_in_domain",
    # Derived ratios
    "urgency_density", "financial_density", "threat_density",
    "caps_word_ratio",
]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 36 handcrafted features from raw text.
    Compatible with sklearn Pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(t) for t in X], dtype=np.float32)

    def _extract(self, text: str) -> list:
        t_lower = text.lower()
        words = text.split()
        n_words = max(len(words), 1)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = max(len(sentences), 1)
        urls = extract_urls(text)

        # ── Keyword hits ──
        urgency_hits     = sum(1 for kw in URGENCY_WORDS if kw in t_lower)
        financial_hits   = sum(1 for kw in FINANCIAL_WORDS if kw in t_lower)
        threat_hits      = sum(1 for kw in THREAT_WORDS if kw in t_lower)
        social_eng_hits  = sum(1 for kw in SOCIAL_ENG_WORDS if kw in t_lower)
        impersonation    = sum(1 for b in IMPERSONATION_BRANDS if b in t_lower)
        manipulation     = sum(1 for p in MANIPULATION_PHRASES if p in t_lower)
        trust_hits       = sum(1 for t in TRUST_INDICATORS if t in t_lower)

        # ── Structural ──
        caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        excl          = text.count("!")
        quest         = text.count("?")
        avg_word_len  = float(np.mean([len(w) for w in words])) if words else 0.0
        avg_sent_len  = float(np.mean([len(s.split()) for s in sentences]))

        # ── Style ──
        double_space  = text.count("  ")
        dollar_signs  = text.count("$")
        number_blobs  = len(re.findall(r'\b\d{3,}\b', text))
        all_caps_w    = sum(1 for w in words if w.isupper() and len(w) > 2)
        ellipsis      = text.count("...")
        commas        = text.count(",")

        # ── URL features ──
        has_url = int(len(urls) > 0)
        domain_risks = []
        url_entropies = []
        has_http_only = 0
        has_shortener = 0
        has_ip = 0
        has_sus_tld = 0
        has_brand_domain = 0
        longest_url = 0

        for u in urls:
            parsed = urlparse(u)
            domain = parsed.netloc.lower()
            domain_risks.append(domain_risk_score(domain))
            url_entropies.append(char_entropy(domain))
            if u.startswith("http://"):
                has_http_only = 1
            if any(s in domain for s in URL_SHORTENERS):
                has_shortener = 1
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                has_ip = 1
            if any(tld in domain for tld in SUSPICIOUS_TLDS):
                has_sus_tld = 1
            for brand in IMPERSONATION_BRANDS:
                if brand.replace(' ', '') in domain:
                    has_brand_domain = 1
            longest_url = max(longest_url, len(u))

        max_domain_risk = max(domain_risks) if domain_risks else 0.0
        avg_domain_risk = float(np.mean(domain_risks)) if domain_risks else 0.0
        avg_url_ent     = float(np.mean(url_entropies)) if url_entropies else 0.0

        # ── Derived ratios ──
        urgency_density  = urgency_hits / n_words
        financial_density = financial_hits / n_words
        threat_density   = threat_hits / n_words
        caps_word_ratio  = all_caps_w / n_words

        return [
            urgency_hits, financial_hits, threat_hits,
            social_eng_hits, impersonation, manipulation, trust_hits,
            caps_ratio, excl, quest,
            len(urls), n_words, len(text), avg_word_len,
            n_sentences, avg_sent_len,
            double_space, dollar_signs, number_blobs,
            all_caps_w, ellipsis, commas,
            has_url, max_domain_risk, avg_domain_risk,
            has_http_only, has_shortener, has_ip,
            has_sus_tld, avg_url_ent, longest_url,
            has_brand_domain,
            urgency_density, financial_density, threat_density,
            caps_word_ratio,
        ]


def analyze_urls(text: str) -> list:
    """Return per-URL risk details for the report."""
    results = []
    for url in extract_urls(text):
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        flags = []
        score = 0

        if any(tld in domain for tld in SUSPICIOUS_TLDS):
            flags.append("Suspicious TLD"); score += 35
        if url.startswith("http://"):
            flags.append("No HTTPS encryption"); score += 20
        if any(s in domain for s in URL_SHORTENERS):
            flags.append("URL shortener (hides destination)"); score += 25
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            flags.append("IP address used as domain"); score += 40
        if domain.count('-') >= 3:
            flags.append("Excessive hyphens in domain"); score += 15
        if domain.count('.') >= 4:
            flags.append("Deep subdomain nesting"); score += 15
        if len(url) > 100:
            flags.append("Unusually long URL"); score += 10
        ent = char_entropy(domain)
        if ent > 3.6:
            flags.append(f"High domain entropy ({ent:.2f}) — looks randomly generated"); score += 20
        for brand in IMPERSONATION_BRANDS:
            if brand.replace(' ', '') in domain:
                flags.append(f"Brand name '{brand}' used in domain (impersonation risk)"); score += 35

        results.append({
            "url":    url,
            "domain": domain,
            "score":  min(100, score),
            "flags":  flags if flags else ["No suspicious indicators found"],
            "safe":   score < 25,
        })
    return results


if __name__ == "__main__":
    # Quick test
    fe = FeatureExtractor()
    tests = [
        "Hey, are we meeting for coffee tomorrow?",
        "URGENT: Your PayPal account is suspended! Verify at http://paypal-secure.tk/verify NOW!!!",
    ]
    for t in tests:
        vec = fe._extract(t)
        print(f"\n[{t[:60]}...]")
        for name, val in zip(FEATURE_NAMES, vec):
            if val != 0:
                print(f"  {name:30s} = {val:.4f}")
