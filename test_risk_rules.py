from __future__ import annotations

import pytest

from risk_rules import label_risk, score_transaction


def base_tx(**overrides):
    tx = {
        "device_risk_score": 0,
        "is_international": 0,
        "amount_usd": 0,
        "velocity_24h": 0,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


class TestDeviceRisk:
    def test_high_device_risk_adds_25(self):
        assert score_transaction(base_tx(device_risk_score=70)) == 25

    def test_device_risk_just_below_high_threshold_adds_10(self):
        assert score_transaction(base_tx(device_risk_score=69)) == 10

    def test_medium_device_risk_adds_10(self):
        assert score_transaction(base_tx(device_risk_score=40)) == 10

    def test_low_device_risk_adds_nothing(self):
        assert score_transaction(base_tx(device_risk_score=39)) == 0


class TestInternational:
    def test_international_adds_15(self):
        assert score_transaction(base_tx(is_international=1)) == 15

    def test_domestic_adds_nothing(self):
        assert score_transaction(base_tx(is_international=0)) == 0


class TestAmount:
    def test_large_amount_adds_25(self):
        assert score_transaction(base_tx(amount_usd=1000)) == 25

    def test_medium_amount_adds_10(self):
        assert score_transaction(base_tx(amount_usd=500)) == 10

    def test_small_amount_adds_nothing(self):
        assert score_transaction(base_tx(amount_usd=499)) == 0


class TestVelocity:
    def test_high_velocity_adds_20(self):
        assert score_transaction(base_tx(velocity_24h=6)) == 20

    def test_velocity_just_below_high_threshold_adds_5(self):
        assert score_transaction(base_tx(velocity_24h=5)) == 5

    def test_medium_velocity_adds_5(self):
        assert score_transaction(base_tx(velocity_24h=3)) == 5

    def test_low_velocity_adds_nothing(self):
        assert score_transaction(base_tx(velocity_24h=2)) == 0


class TestFailedLogins:
    def test_many_failed_logins_adds_20(self):
        assert score_transaction(base_tx(failed_logins_24h=5)) == 20

    def test_failed_logins_just_below_high_threshold_adds_10(self):
        assert score_transaction(base_tx(failed_logins_24h=4)) == 10

    def test_some_failed_logins_adds_10(self):
        assert score_transaction(base_tx(failed_logins_24h=2)) == 10

    def test_few_failed_logins_adds_nothing(self):
        assert score_transaction(base_tx(failed_logins_24h=1)) == 0


class TestPriorChargebacks:
    def test_multiple_chargebacks_adds_20(self):
        assert score_transaction(base_tx(prior_chargebacks=2)) == 20

    def test_one_chargeback_adds_5(self):
        assert score_transaction(base_tx(prior_chargebacks=1)) == 5

    def test_no_chargebacks_adds_nothing(self):
        assert score_transaction(base_tx(prior_chargebacks=0)) == 0


class TestScoreBounds:
    def test_score_clamped_at_100(self):
        tx = base_tx(
            device_risk_score=70,
            is_international=1,
            amount_usd=1000,
            velocity_24h=6,
            failed_logins_24h=5,
            prior_chargebacks=2,
        )
        assert score_transaction(tx) == 100

    def test_clean_transaction_scores_zero(self):
        assert score_transaction(base_tx()) == 0


class TestLabelRisk:
    def test_score_60_is_high(self):
        assert label_risk(60) == "high"

    def test_score_100_is_high(self):
        assert label_risk(100) == "high"

    def test_score_30_is_medium(self):
        assert label_risk(30) == "medium"

    def test_score_59_is_medium(self):
        assert label_risk(59) == "medium"

    def test_score_0_is_low(self):
        assert label_risk(0) == "low"

    def test_score_29_is_low(self):
        assert label_risk(29) == "low"
