from __future__ import annotations

import pandas as pd
import pytest

from analyze_fraud import score_transactions, summarize_results
from features import build_model_frame


# ── shared helpers ────────────────────────────────────────────────────────────

def make_transactions(**overrides):
    row = {
        "transaction_id": 1,
        "account_id": 100,
        "amount_usd": 0.0,
        "device_risk_score": 0,
        "is_international": 0,
        "velocity_24h": 0,
        "failed_logins_24h": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def make_accounts(**overrides):
    row = {"account_id": 100, "prior_chargebacks": 0}
    row.update(overrides)
    return pd.DataFrame([row])


def make_scored(rows):
    return pd.DataFrame(rows, columns=["transaction_id", "risk_label", "amount_usd"])


def make_chargebacks(ids):
    return pd.DataFrame({"transaction_id": ids})


# ── build_model_frame ─────────────────────────────────────────────────────────

class TestBuildModelFrame:
    def test_merges_account_columns(self):
        result = build_model_frame(make_transactions(), make_accounts(prior_chargebacks=3))
        assert result["prior_chargebacks"].iloc[0] == 3

    def test_is_large_amount_set_at_threshold(self):
        result = build_model_frame(make_transactions(amount_usd=1000.0), make_accounts())
        assert result["is_large_amount"].iloc[0] == 1

    def test_is_large_amount_not_set_below_threshold(self):
        result = build_model_frame(make_transactions(amount_usd=999.99), make_accounts())
        assert result["is_large_amount"].iloc[0] == 0

    def test_login_pressure_none_for_zero_failures(self):
        result = build_model_frame(make_transactions(failed_logins_24h=0), make_accounts())
        assert result["login_pressure"].iloc[0] == "none"

    def test_login_pressure_low_for_one_failure(self):
        result = build_model_frame(make_transactions(failed_logins_24h=1), make_accounts())
        assert result["login_pressure"].iloc[0] == "low"

    def test_login_pressure_low_for_two_failures(self):
        result = build_model_frame(make_transactions(failed_logins_24h=2), make_accounts())
        assert result["login_pressure"].iloc[0] == "low"

    def test_login_pressure_high_for_three_failures(self):
        result = build_model_frame(make_transactions(failed_logins_24h=3), make_accounts())
        assert result["login_pressure"].iloc[0] == "high"

    def test_unknown_account_produces_null_fields(self):
        txns = make_transactions(account_id=999)
        accts = make_accounts(account_id=100)
        result = build_model_frame(txns, accts)
        assert pd.isna(result["prior_chargebacks"].iloc[0])


# ── score_transactions ────────────────────────────────────────────────────────

class TestScoreTransactions:
    def test_output_has_risk_score_and_label_columns(self):
        result = score_transactions(make_transactions(), make_accounts())
        assert "risk_score" in result.columns
        assert "risk_label" in result.columns

    def test_high_risk_transaction_labeled_high(self):
        txns = make_transactions(
            device_risk_score=80,
            is_international=1,
            amount_usd=1500.0,
            velocity_24h=8,
            failed_logins_24h=6,
        )
        result = score_transactions(txns, make_accounts(prior_chargebacks=3))
        assert result["risk_score"].iloc[0] == 100
        assert result["risk_label"].iloc[0] == "high"

    def test_clean_transaction_labeled_low(self):
        result = score_transactions(make_transactions(), make_accounts())
        assert result["risk_score"].iloc[0] == 0
        assert result["risk_label"].iloc[0] == "low"


# ── summarize_results ─────────────────────────────────────────────────────────

class TestSummarizeResults:
    def test_chargeback_rate_is_1_when_all_charged_back(self):
        scored = make_scored([(1, "high", 500.0), (2, "high", 1000.0)])
        result = summarize_results(scored, make_chargebacks([1, 2]))
        row = result[result["risk_label"] == "high"].iloc[0]
        assert row["chargeback_rate"] == 1.0

    def test_chargeback_rate_is_0_when_no_chargebacks(self):
        scored = make_scored([(1, "low", 50.0), (2, "low", 75.0)])
        result = summarize_results(scored, make_chargebacks([999]))
        row = result[result["risk_label"] == "low"].iloc[0]
        assert row["chargeback_rate"] == 0.0
        assert row["chargebacks"] == 0

    def test_chargeback_rate_partial(self):
        scored = make_scored([(1, "high", 200.0), (2, "high", 200.0)])
        result = summarize_results(scored, make_chargebacks([1]))
        row = result[result["risk_label"] == "high"].iloc[0]
        assert row["chargeback_rate"] == 0.5
        assert row["chargebacks"] == 1

    def test_transaction_count(self):
        scored = make_scored([(1, "medium", 100.0), (2, "medium", 200.0), (3, "medium", 300.0)])
        result = summarize_results(scored, make_chargebacks([999]))
        row = result[result["risk_label"] == "medium"].iloc[0]
        assert row["transactions"] == 3

    def test_total_amount_usd(self):
        scored = make_scored([(1, "high", 300.0), (2, "high", 700.0)])
        result = summarize_results(scored, make_chargebacks([999]))
        row = result[result["risk_label"] == "high"].iloc[0]
        assert row["total_amount_usd"] == pytest.approx(1000.0)

    def test_avg_amount_usd(self):
        scored = make_scored([(1, "high", 200.0), (2, "high", 600.0)])
        result = summarize_results(scored, make_chargebacks([999]))
        row = result[result["risk_label"] == "high"].iloc[0]
        assert row["avg_amount_usd"] == pytest.approx(400.0)

    def test_multiple_risk_labels_reported_separately(self):
        scored = make_scored([(1, "high", 1000.0), (2, "low", 50.0)])
        result = summarize_results(scored, make_chargebacks([1]))
        assert len(result) == 2
        assert set(result["risk_label"]) == {"high", "low"}
        high_row = result[result["risk_label"] == "high"].iloc[0]
        low_row = result[result["risk_label"] == "low"].iloc[0]
        assert high_row["chargeback_rate"] == 1.0
        assert low_row["chargeback_rate"] == 0.0
