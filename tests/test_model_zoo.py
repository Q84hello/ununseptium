"""Tests for model_zoo module."""

from __future__ import annotations

from ununseptium.model_zoo.catalog import (
    ModelArchitecture,
    ModelCatalog,
    ModelDomain,
    ModelEntry,
)


class TestModelCatalog:
    """Test ModelCatalog."""

    def test_catalog_creation(self):
        """Test catalog instantiation."""
        catalog = ModelCatalog()
        assert catalog is not None
        assert len(catalog) > 0  # Has built-in models

    def test_get_model(self):
        """Test getting model by ID."""
        catalog = ModelCatalog()

        model = catalog.get("aml-transaction-risk-v1")

        assert model is not None
        assert model.model_id == "aml-transaction-risk-v1"
        assert model.domain == ModelDomain.AML

    def test_get_nonexistent_model(self):
        """Test getting non-existent model."""
        catalog = ModelCatalog()

        model = catalog.get("nonexistent-model")

        assert model is None

    def test_list_all_models(self):
        """Test listing all models."""
        catalog = ModelCatalog()

        models = catalog.list_models()

        assert len(models) > 0
        assert all(isinstance(m, ModelEntry) for m in models)

    def test_list_by_domain(self):
        """Test listing models by domain."""
        catalog = ModelCatalog()

        aml_models = catalog.list_models(domain=ModelDomain.AML)
        kyc_models = catalog.list_models(domain=ModelDomain.KYC)

        assert len(aml_models) > 0
        assert all(m.domain == ModelDomain.AML for m in aml_models)
        assert len(kyc_models) > 0
        assert all(m.domain == ModelDomain.KYC for m in kyc_models)

    def test_list_by_architecture(self):
        """Test listing models by architecture."""
        catalog = ModelCatalog()

        gb_models = catalog.list_models(architecture=ModelArchitecture.GRADIENT_BOOSTING)

        assert len(gb_models) > 0
        assert all(m.architecture == ModelArchitecture.GRADIENT_BOOSTING for m in gb_models)

    def test_list_by_tag(self):
        """Test listing models by tag."""
        catalog = ModelCatalog()

        aml_tagged = catalog.list_models(tag="aml")

        assert len(aml_tagged) > 0
        assert all("aml" in m.tags for m in aml_tagged)

    def test_search_models(self):
        """Test searching models."""
        catalog = ModelCatalog()

        results = catalog.search("transaction")

        assert len(results) > 0
        assert any(
            "transaction" in m.name.lower() or "transaction" in m.description.lower()
            for m in results
        )

    def test_search_no_results(self):
        """Test search with no matches."""
        catalog = ModelCatalog()

        results = catalog.search("xyznotfound")

        assert len(results) == 0

    def test_register_custom_model(self):
        """Test registering custom model."""
        catalog = ModelCatalog()
        initial_count = len(catalog)

        custom_model = ModelEntry(
            model_id="custom-test-model",
            name="Custom Test Model",
            version="1.0.0",
            domain=ModelDomain.RISK,
        )
        catalog.register(custom_model)

        assert len(catalog) == initial_count + 1
        retrieved = catalog.get("custom-test-model")
        assert retrieved is not None
        assert retrieved.name == "Custom Test Model"

    def test_builtin_models_have_metadata(self):
        """Test that built-in models have complete metadata."""
        catalog = ModelCatalog()

        model = catalog.get("aml-transaction-risk-v1")

        assert model is not None
        assert model.name
        assert model.version
        assert model.description
        assert len(model.metrics) > 0
        assert len(model.input_features) > 0
        assert len(model.tags) > 0

    def test_model_entry_creation(self):
        """Test creating ModelEntry."""
        entry = ModelEntry(
            model_id="test-model",
            name="Test Model",
            version="2.0.0",
            domain=ModelDomain.FRAUD,
            architecture=ModelArchitecture.NEURAL_NETWORK,
        )

        assert entry.model_id == "test-model"
        assert entry.name == "Test Model"
        assert entry.version == "2.0.0"
        assert entry.domain == ModelDomain.FRAUD
        assert entry.architecture == ModelArchitecture.NEURAL_NETWORK
