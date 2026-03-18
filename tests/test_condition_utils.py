from __future__ import annotations

import anndata
import anndata as ad
import numpy as np
import pandas as pd
import pytest

N_CELLS = 80
N_GENES = 15
N_DONORS = 8


@pytest.fixture
def base_adata():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 200, size=(N_CELLS, N_GENES)).astype("float32")
    obs = pd.DataFrame(
        {
            "donor_id": [f"donor_{i % N_DONORS:02d}" for i in range(N_CELLS)],
            "condition": ["ctrl"] * (N_CELLS // 2) + ["treat"] * (N_CELLS // 2),
            "subtype": ["A"] * (N_CELLS // 2) + ["B"] * (N_CELLS // 2),
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)])
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def three_condition_adata(base_adata):
    """AnnData with a 3-level condition to produce multiple contrasts."""
    extra = base_adata[:10].copy()
    extra.obs["condition"] = "extra"
    return anndata.concat([base_adata, extra])


class _FakeModel:
    """Mimics the minimal pertpy DE interface."""

    _last_kwargs: dict = {}

    @classmethod
    def compare_groups(cls, adata, column, baseline, groups_to_compare, **kwargs):
        cls._last_kwargs = kwargs
        rows = [
            {"variable": f"gene_{i}", "log_fc": i * 0.1, "p_value": 0.01 * (i + 1), "adj_p_value": 0.05 * (i + 1)}
            for i in range(3)
        ]
        return pd.DataFrame(rows)


class _FailingModel:
    @classmethod
    def compare_groups(cls, adata, column, baseline, groups_to_compare, **kwargs):
        raise RuntimeError("Deliberate failure")


class TestBuildConditionCombinations:
    def test_returns_dataframe(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        assert isinstance(build_condition_combinations(base_adata, ["condition", "subtype"]), pd.DataFrame)

    def test_has_label_column(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        assert "label" in build_condition_combinations(base_adata, ["condition", "subtype"]).columns

    def test_labels_use_sep(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        result = build_condition_combinations(base_adata, ["condition", "subtype"], sep="-")
        assert result["label"].str.contains("-").all()

    def test_only_observed_combinations(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        # ctrl_A and treat_B only — no ctrl_B or treat_A in the fixture
        result = build_condition_combinations(base_adata, ["condition", "subtype"])
        assert len(result) == 2

    def test_missing_column_raises(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        with pytest.raises(ValueError, match="not found"):
            build_condition_combinations(base_adata, ["condition", "nonexistent"])

    def test_empty_cols_raises(self, base_adata):
        from patpy.tl.condition_utils import build_condition_combinations

        with pytest.raises(ValueError):
            build_condition_combinations(base_adata, [])


class TestBuildAllPairwiseContrasts:
    def test_returns_list_of_dicts(self, base_adata):
        from patpy.tl.condition_utils import build_all_pairwise_contrasts

        result = build_all_pairwise_contrasts(base_adata, ["condition", "subtype"])
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_each_contrast_has_required_keys(self, base_adata):
        from patpy.tl.condition_utils import build_all_pairwise_contrasts

        for c in build_all_pairwise_contrasts(base_adata, ["condition", "subtype"]):
            assert {"group", "baseline", "label"} == set(c.keys())

    def test_label_contains_vs(self, base_adata):
        from patpy.tl.condition_utils import build_all_pairwise_contrasts

        for c in build_all_pairwise_contrasts(base_adata, ["condition", "subtype"]):
            assert "_vs_" in c["label"]

    def test_n_contrasts_two_combos(self, base_adata):
        from patpy.tl.condition_utils import build_all_pairwise_contrasts

        # 2 observed combos → C(2,2)=1 pairwise contrast
        assert len(build_all_pairwise_contrasts(base_adata, ["condition", "subtype"])) == 1


class TestAddCombinedConditionColumn:
    def test_adds_default_column(self, base_adata):
        from patpy.tl.condition_utils import add_combined_condition_column

        add_combined_condition_column(base_adata, ["condition", "subtype"])
        assert "condition_combined" in base_adata.obs.columns

    def test_custom_col_name(self, base_adata):
        from patpy.tl.condition_utils import add_combined_condition_column

        add_combined_condition_column(base_adata, ["condition", "subtype"], new_col="tp_sub")
        assert "tp_sub" in base_adata.obs.columns

    def test_values_use_sep(self, base_adata):
        from patpy.tl.condition_utils import add_combined_condition_column

        add_combined_condition_column(base_adata, ["condition", "subtype"], sep="|")
        assert base_adata.obs["condition_combined"].str.contains("|", regex=False).all()


class TestFilterAdataToConditions:
    def test_filters_correctly(self, base_adata):
        from patpy.tl.condition_utils import (
            add_combined_condition_column,
            filter_adata_to_conditions,
        )

        add_combined_condition_column(base_adata, ["condition", "subtype"])
        sub = filter_adata_to_conditions(base_adata, "condition_combined", ["ctrl_A"])
        assert (sub.obs["condition_combined"] == "ctrl_A").all()

    def test_returns_copy(self, base_adata):
        from patpy.tl.condition_utils import filter_adata_to_conditions

        sub = filter_adata_to_conditions(base_adata, "condition", ["ctrl"])
        sub.obs["condition"] = "modified"
        assert (base_adata.obs["condition"] != "modified").any()


class TestRunConditionCombinations:
    def test_returns_dataframe(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        res = run_condition_combinations(_FakeModel, base_adata, ["condition", "subtype"])
        assert isinstance(res, pd.DataFrame)

    def test_has_contrast_column(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        res = run_condition_combinations(_FakeModel, base_adata, ["condition", "subtype"])
        assert "contrast" in res.columns

    def test_contrast_label_format(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        res = run_condition_combinations(_FakeModel, base_adata, ["condition", "subtype"])
        assert res["contrast"].str.contains("_vs_").all()

    def test_adds_combined_column_to_adata(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        run_condition_combinations(_FakeModel, base_adata, ["condition", "subtype"], combined_col="tp_sub")
        assert "tp_sub" in base_adata.obs.columns

    def test_custom_sep(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        run_condition_combinations(_FakeModel, base_adata, ["condition", "subtype"], sep="-")
        assert base_adata.obs["condition_combined"].str.contains("-").any()

    def test_kwargs_forwarded_to_model(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        run_condition_combinations(
            _FakeModel,
            base_adata,
            ["condition", "subtype"],
            layer="counts",
            paired_by="donor_id",
        )
        assert _FakeModel._last_kwargs.get("layer") == "counts"
        assert _FakeModel._last_kwargs.get("paired_by") == "donor_id"

    def test_subset_contrasts_limits_run(self, base_adata, three_condition_adata):
        from patpy.tl.condition_utils import (
            build_all_pairwise_contrasts,
            run_condition_combinations,
        )

        all_c = build_all_pairwise_contrasts(three_condition_adata, ["condition", "subtype"])
        subset = all_c[:1]
        res = run_condition_combinations(
            _FakeModel,
            three_condition_adata,
            ["condition", "subtype"],
            subset_contrasts=subset,
        )
        assert res["contrast"].nunique() == 1
        assert res["contrast"].iloc[0] == subset[0]["label"]

    def test_failed_contrast_skipped_with_warning(self, three_condition_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        call_count = [0]
        original = _FakeModel.compare_groups

        @classmethod
        def _fail_once(cls, adata, column, baseline, groups_to_compare, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first call fails")
            return original.__func__(cls, adata, column, baseline, groups_to_compare, **kwargs)

        _FakeModel.compare_groups = _fail_once
        try:
            with pytest.warns(UserWarning, match="failed"):
                res = run_condition_combinations(_FakeModel, three_condition_adata, ["condition", "subtype"])
            assert isinstance(res, pd.DataFrame)
        finally:
            _FakeModel.compare_groups = original

    def test_all_contrasts_fail_raises(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        with pytest.warns(UserWarning, match="failed"):
            with pytest.raises(RuntimeError, match="All contrasts failed"):
                run_condition_combinations(_FailingModel, base_adata, ["condition", "subtype"])

    def test_single_condition_col(self, base_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        res = run_condition_combinations(_FakeModel, base_adata, ["condition"])
        assert isinstance(res, pd.DataFrame)
        assert "contrast" in res.columns

    def test_multiple_contrasts_concatenated(self, three_condition_adata):
        from patpy.tl.condition_utils import run_condition_combinations

        res = run_condition_combinations(_FakeModel, three_condition_adata, ["condition", "subtype"])
        assert res["contrast"].nunique() > 1


class TestConditionComparison:
    def test_stores_model_and_kwargs(self):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel, layer="counts", paired_by="donor_id")
        assert cc.model_cls is _FakeModel
        assert cc.default_kwargs["layer"] == "counts"
        assert cc.default_kwargs["paired_by"] == "donor_id"

    def test_run_returns_dataframe(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        res = ConditionComparison(_FakeModel).run(base_adata, ["condition", "subtype"])
        assert isinstance(res, pd.DataFrame)
        assert "contrast" in res.columns

    def test_default_kwargs_forwarded(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        ConditionComparison(_FakeModel, layer="my_layer").run(base_adata, ["condition", "subtype"])
        assert _FakeModel._last_kwargs.get("layer") == "my_layer"

    def test_per_call_kwargs_override_defaults(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel, layer="default_layer")
        cc.run(base_adata, ["condition", "subtype"], layer="override_layer")
        assert _FakeModel._last_kwargs.get("layer") == "override_layer"

    def test_per_call_kwargs_extend_defaults(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel, layer="counts")
        cc.run(base_adata, ["condition", "subtype"], paired_by="donor_id")
        assert _FakeModel._last_kwargs.get("layer") == "counts"
        assert _FakeModel._last_kwargs.get("paired_by") == "donor_id"

    def test_repr_contains_class_name(self):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel, layer="counts")
        assert "_FakeModel" in repr(cc)
        assert "counts" in repr(cc)

    def test_run_with_subset_contrasts(self, base_adata, three_condition_adata):
        from patpy.tl.condition_utils import (
            ConditionComparison,
            build_all_pairwise_contrasts,
        )

        all_c = build_all_pairwise_contrasts(three_condition_adata, ["condition", "subtype"])
        res = ConditionComparison(_FakeModel).run(
            three_condition_adata, ["condition", "subtype"], subset_contrasts=all_c[:1]
        )
        assert res["contrast"].nunique() == 1

    def test_reuse_across_datasets(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel, layer="counts")
        res1 = cc.run(base_adata, ["condition", "subtype"])
        res2 = cc.run(base_adata.copy(), ["condition"])
        assert isinstance(res1, pd.DataFrame)
        assert isinstance(res2, pd.DataFrame)

    def test_no_default_kwargs(self, base_adata):
        from patpy.tl.condition_utils import ConditionComparison

        cc = ConditionComparison(_FakeModel)
        assert cc.default_kwargs == {}
        res = cc.run(base_adata, ["condition", "subtype"])
        assert isinstance(res, pd.DataFrame)

