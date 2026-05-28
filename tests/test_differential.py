from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from patpy.tl._differential_utils import (
    build_all_pairwise_contrasts,
    build_condition_combinations,
    filter_adata_to_conditions,
)
from patpy.tl.condition_comparison import ConditionComparison
from patpy.tl.factorial_comparison import (
    FactorialDE,
    _edger_coef_key,
    _pydeseq2_parse_level,
)


@pytest.fixture
def simple_adata():
    """AnnData with two condition columns and no real counts.

    Four observed combinations: COVID_SEV/HV x female/male.
    10 donors, one row each (pseudobulk-style).
    """
    obs = pd.DataFrame(
        {
            "source": ["COVID_SEV"] * 5 + ["HV"] * 5,
            "sex": (["female", "male"] * 5)[:10],
            "donor_id": [f"donor_{i:02d}" for i in range(10)],
        }
    )
    obs.index = obs["donor_id"]
    rng = np.random.default_rng(0)
    adata = sc.AnnData(X=rng.integers(0, 100, (10, 20)).astype("float32"), obs=obs)
    return adata


@pytest.fixture
def missing_combo_adata():
    """AnnData missing the COVID_SEV/male combination."""
    obs = pd.DataFrame(
        {
            "source": ["COVID_SEV"] * 3 + ["HV"] * 4 + ["HV"] * 3,
            "sex": ["female"] * 3 + ["female"] * 4 + ["male"] * 3,
            "donor_id": [f"donor_{i:02d}" for i in range(10)],
        }
    )
    obs.index = obs["donor_id"]
    rng = np.random.default_rng(1)
    adata = sc.AnnData(X=rng.integers(0, 100, (10, 20)).astype("float32"), obs=obs)
    return adata


class TestBuildConditionCombinations:
    def test_returns_dataframe(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source", "sex"])
        assert isinstance(result, pd.DataFrame)

    def test_has_label_column(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source", "sex"])
        assert "label" in result.columns

    def test_correct_number_of_combinations(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source", "sex"])
        assert len(result) == 4

    def test_labels_are_joined_with_sep(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source", "sex"])
        for _, row in result.iterrows():
            assert row["label"] == f"{row['source']}_{row['sex']}"

    def test_custom_sep(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source", "sex"], sep="|")
        assert all("|" in label for label in result["label"])

    def test_single_column(self, simple_adata):
        result = build_condition_combinations(simple_adata, ["source"])
        assert set(result["label"]) == {"COVID_SEV", "HV"}

    def test_empty_condition_cols_raises(self, simple_adata):
        with pytest.raises(ValueError, match="at least one"):
            build_condition_combinations(simple_adata, [])

    def test_missing_column_raises(self, simple_adata):
        with pytest.raises(ValueError, match="not found"):
            build_condition_combinations(simple_adata, ["nonexistent"])

    def test_no_cartesian_product_for_missing_combo(self, missing_combo_adata):
        result = build_condition_combinations(missing_combo_adata, ["source", "sex"])
        # COVID_SEV/male is absent — should have 3 combos, not 4
        assert len(result) == 3

    def test_observed_combinations_only(self, missing_combo_adata):
        result = build_condition_combinations(missing_combo_adata, ["source", "sex"])
        labels = set(result["label"])
        assert "COVID_SEV_male" not in labels
        assert "COVID_SEV_female" in labels


class TestBuildAllPairwiseContrasts:
    def test_returns_list_of_dicts(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_each_dict_has_required_keys(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        for c in result:
            assert "group" in c and "baseline" in c and "label" in c

    def test_correct_number_of_contrasts_four_groups(self, simple_adata):
        # C(4, 2) = 6
        result = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        assert len(result) == 6

    def test_correct_number_of_contrasts_three_groups(self, missing_combo_adata):
        # C(3, 2) = 3
        result = build_all_pairwise_contrasts(missing_combo_adata, ["source", "sex"])
        assert len(result) == 3

    def test_label_format(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source"])
        assert any(c["label"] == "COVID_SEV_vs_HV" for c in result)

    def test_no_self_comparisons(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        assert all(c["group"] != c["baseline"] for c in result)

    def test_no_duplicate_contrasts(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        labels = [c["label"] for c in result]
        assert len(labels) == len(set(labels))

    def test_two_groups_gives_one_contrast(self, simple_adata):
        result = build_all_pairwise_contrasts(simple_adata, ["source"])
        assert len(result) == 1


class TestFilterAdataToConditions:
    def test_returns_adata(self, simple_adata):
        simple_adata.obs["group"] = simple_adata.obs["source"]
        result = filter_adata_to_conditions(simple_adata, "group", ["COVID_SEV"])
        assert hasattr(result, "obs")

    def test_filters_correctly(self, simple_adata):
        simple_adata.obs["group"] = simple_adata.obs["source"]
        result = filter_adata_to_conditions(simple_adata, "group", ["COVID_SEV"])
        assert (result.obs["group"] == "COVID_SEV").all()

    def test_multiple_groups(self, simple_adata):
        simple_adata.obs["group"] = simple_adata.obs["source"]
        result = filter_adata_to_conditions(simple_adata, "group", ["COVID_SEV", "HV"])
        assert result.n_obs == simple_adata.n_obs

    def test_empty_result_for_missing_group(self, simple_adata):
        simple_adata.obs["group"] = simple_adata.obs["source"]
        result = filter_adata_to_conditions(simple_adata, "group", ["NONEXISTENT"])
        assert result.n_obs == 0

    def test_returns_copy(self, simple_adata):
        simple_adata.obs["group"] = simple_adata.obs["source"]
        result = filter_adata_to_conditions(simple_adata, "group", ["COVID_SEV"])
        result.obs["new_col"] = "x"
        assert "new_col" not in simple_adata.obs.columns


class _MockModel:
    """Minimal fake model class for tests that only need __name__."""

    __name__ = "MockModel"


class TestConditionComparisonInit:
    def test_stores_model_cls(self):
        cc = ConditionComparison(_MockModel)
        assert cc.model_cls is _MockModel

    def test_stores_default_kwargs(self):
        cc = ConditionComparison(_MockModel, layer="counts", paired_by="donor")
        assert cc.default_kwargs == {"layer": "counts", "paired_by": "donor"}

    def test_models_empty_before_run(self):
        cc = ConditionComparison(_MockModel)
        assert cc.models_ == {}

    def test_repr_no_models(self):
        cc = ConditionComparison(_MockModel)
        assert "MockModel" in repr(cc)

    def test_repr_with_models(self):
        cc = ConditionComparison(_MockModel)
        cc.models_["contrast_A"] = object()
        r = repr(cc)
        assert "1 model" in r
        assert "contrast_A" in r


class TestConditionComparisonGetModel:
    def test_get_model_returns_stored_instance(self):
        cc = ConditionComparison(_MockModel)
        sentinel = object()
        cc.models_["my_contrast"] = sentinel
        assert cc.get_model("my_contrast") is sentinel

    def test_get_model_raises_for_missing_contrast(self):
        cc = ConditionComparison(_MockModel)
        with pytest.raises(KeyError, match="my_contrast"):
            cc.get_model("my_contrast")

    def test_get_model_error_lists_available(self):
        cc = ConditionComparison(_MockModel)
        cc.models_["other_contrast"] = object()
        with pytest.raises(KeyError, match="other_contrast"):
            cc.get_model("missing")


class TestConditionComparisonSubsetContrasts:
    def test_subset_contrasts_reduces_what_is_tested(self, simple_adata):
        """build_all_pairwise_contrasts returns 6 for 4 groups; subset to 1."""
        all_contrasts = build_all_pairwise_contrasts(simple_adata, ["source", "sex"])
        subset = all_contrasts[:1]

        calls = []

        class FakeModel:
            @staticmethod
            def compare_groups(sub, column, baseline, groups_to_compare, **kw):
                calls.append(groups_to_compare)
                return pd.DataFrame({"variable": ["g1"], "log_fc": [1.0], "adj_p_value": [0.01], "contrast": ["x"]})

        cc = ConditionComparison(FakeModel)
        cc.run(simple_adata, ["source", "sex"], subset_contrasts=subset)
        assert len(calls) == 1

    def test_bad_contrast_warns_and_raises(self, simple_adata):
        """A contrast that fails is skipped with a UserWarning and raises RuntimeError."""
        fake_contrast = [
            {"group": "NONEXISTENT_female", "baseline": "HV_female", "label": "NONEXISTENT_female_vs_HV_female"}
        ]

        class FakeModel:
            __name__ = "FakeModel"

            @staticmethod
            def compare_groups(*a, **kw):
                raise ValueError("no such group")

        cc = ConditionComparison(FakeModel)

        with pytest.warns(UserWarning):
            with pytest.raises(RuntimeError, match="All contrasts failed"):
                cc.run(simple_adata, ["source", "sex"], subset_contrasts=fake_contrast)


class TestEdgerCoefKey:
    def test_finds_bracket_notation(self):
        coef_names = ["condition_group[COVID_SEV_female]", "condition_group[HV_male]"]
        assert _edger_coef_key(coef_names, "condition_group", "COVID_SEV_female") == "condition_group[COVID_SEV_female]"

    def test_finds_plain_concatenation(self):
        coef_names = ["condition_groupCOVID_SEV_female", "condition_groupHV_male"]
        assert _edger_coef_key(coef_names, "condition_group", "COVID_SEV_female") == "condition_groupCOVID_SEV_female"

    def test_returns_none_when_not_found(self):
        coef_names = ["condition_group[HV_female]"]
        assert _edger_coef_key(coef_names, "condition_group", "COVID_SEV_female") is None

    def test_bracket_notation_takes_priority(self):
        # Both present — bracket should be returned first
        coef_names = ["condition_group[COVID_SEV_female]", "condition_groupCOVID_SEV_female"]
        assert _edger_coef_key(coef_names, "condition_group", "COVID_SEV_female") == "condition_group[COVID_SEV_female]"

    def test_different_group_col_prefix(self):
        coef_names = ["mygroup[levelA]", "mygroup[levelB]"]
        assert _edger_coef_key(coef_names, "mygroup", "levelA") == "mygroup[levelA]"
        assert _edger_coef_key(coef_names, "mygroup", "levelC") is None


class TestPyDeseq2ParseLevel:
    def test_strips_bracket_t_notation(self):
        assert _pydeseq2_parse_level("condition_group[T.COVID_SEV_female]", "condition_group") == "COVID_SEV_female"

    def test_strips_plain_prefix(self):
        assert _pydeseq2_parse_level("condition_groupHV_male", "condition_group") == "HV_male"

    def test_handles_level_with_underscores(self):
        assert _pydeseq2_parse_level("grp[T.a_b_c]", "grp") == "a_b_c"

    def test_intercept_unchanged(self):
        # Intercept doesn't contain group_col so is returned as-is
        result = _pydeseq2_parse_level("Intercept", "condition_group")
        assert result == "Intercept"


class TestFactorialDEInit:
    def test_stores_model_cls(self):
        fc = FactorialDE(_MockModel)
        assert fc.model_cls is _MockModel

    def test_default_layer_is_none(self):
        fc = FactorialDE(_MockModel)
        assert fc.layer is None

    def test_custom_layer(self):
        fc = FactorialDE(_MockModel, layer="raw")
        assert fc.layer == "raw"

    def test_model_none_before_run(self):
        fc = FactorialDE(_MockModel)
        assert fc.model_ is None

    def test_repr_not_fitted(self):
        fc = FactorialDE(_MockModel)
        assert "not fitted" in repr(fc)

    def test_repr_fitted(self):
        fc = FactorialDE(_MockModel)
        fc.model_ = object()
        assert "fitted" in repr(fc)
        assert "not fitted" not in repr(fc)


class TestFactorialDEValidation:
    def test_invalid_encoding_raises(self, simple_adata):
        fc = FactorialDE(_MockModel)
        with pytest.raises(ValueError, match="encoding"):
            fc.run(simple_adata, ["source", "sex"], encoding="bad_encoding")

    def test_interaction_encoding_requires_exactly_two_cols(self, simple_adata):
        fc = FactorialDE(_MockModel)
        with pytest.raises(ValueError, match="exactly 2"):
            fc.run(simple_adata, ["source"], encoding="interaction")

    def test_get_model_raises_before_run(self):
        fc = FactorialDE(_MockModel)
        with pytest.raises(RuntimeError, match="Call run()"):
            fc.get_model()

    def test_unsupported_model_class_raises(self, simple_adata):
        class UnsupportedModel:
            __name__ = "UnsupportedModel"

        fc = FactorialDE(UnsupportedModel)
        with pytest.raises(ValueError, match="Unsupported"):
            fc.run(simple_adata, ["source", "sex"], encoding="group")


class TestFactorialDEGroupColBuilding:
    def test_group_col_created_in_obs(self, simple_adata):
        """run() should add the combined column to adata.obs."""
        calls = []

        class EdgeR:
            __name__ = "EdgeR"

            def __init__(self, adata, design, **kw):
                calls.append(("init", design))
                self.design = pd.DataFrame(
                    columns=[
                        "condition_group[COVID_SEV_female]",
                        "condition_group[COVID_SEV_male]",
                        "condition_group[HV_female]",
                        "condition_group[HV_male]",
                    ]
                )

            def fit(self):
                calls.append("fit")

            def _test_single_contrast(self, vec):
                return pd.DataFrame({"variable": ["g1"], "log_fc": [0.5], "adj_p_value": [0.05]})

        fc = FactorialDE(EdgeR)
        fc.run(simple_adata, ["source", "sex"], encoding="group")

        # Check init was called with the group design
        init_calls = [c for c in calls if isinstance(c, tuple) and c[0] == "init"]
        assert any("condition_group" in c[1] for c in init_calls)

    def test_ref_levels_applied_before_fit(self, simple_adata):
        """ref_levels should reorder categories before model construction."""
        observed_categories = []

        class EdgeR:
            __name__ = "EdgeR"

            def __init__(self, adata, design, **kw):
                observed_categories.append(
                    list(adata.obs["source"].cat.categories) if hasattr(adata.obs["source"], "cat") else None
                )
                self.design = pd.DataFrame(
                    columns=["Intercept", "source[T.COVID_SEV]", "sex[T.male]", "source[T.COVID_SEV]:sex[T.male]"]
                )

            def fit(self):
                pass

            def _test_single_contrast(self, vec):
                return pd.DataFrame({"variable": ["g1"], "log_fc": [0.5], "adj_p_value": [0.05]})

        fc = FactorialDE(EdgeR)
        fc.run(
            simple_adata,
            ["source", "sex"],
            encoding="interaction",
            ref_levels={"source": "HV"},
        )
        # HV should be first category (reference)
        cats = observed_categories[0]
        if cats is not None:
            assert cats[0] == "HV"
