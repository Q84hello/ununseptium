"""Model catalog for pretrained models.

Provides model registry and metadata management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelDomain(str, Enum):
    """Model application domains."""

    AML = "aml"
    KYC = "kyc"
    FRAUD = "fraud"
    RISK = "risk"
    ANOMALY = "anomaly"
    GENERAL = "general"


class ModelArchitecture(str, Enum):
    """Model architecture types."""

    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"
    GRAPH_NEURAL_NETWORK = "gnn"


class ModelEntry(BaseModel):
    """Entry in the model catalog.

    Attributes:
        model_id: Unique model identifier.
        name: Human-readable name.
        version: Model version.
        domain: Application domain.
        architecture: Model architecture.
        description: Model description.
        metrics: Performance metrics.
        input_features: Required input features.
        output_schema: Output schema.
        file_size_mb: Model file size.
        download_url: Download URL.
        checksum: SHA-256 checksum.
        created_at: Creation timestamp.
        tags: Model tags.
        metadata: Additional metadata.
    """

    model_id: str
    name: str
    version: str = "1.0.0"
    domain: ModelDomain = ModelDomain.GENERAL
    architecture: ModelArchitecture = ModelArchitecture.ENSEMBLE
    description: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    input_features: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    file_size_mb: float = 0.0
    download_url: str = ""
    checksum: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCatalog:
    """Catalog of available pretrained models.

    Example:
        ```python
        from ununseptium.model_zoo import ModelCatalog

        catalog = ModelCatalog()

        # List available models
        models = catalog.list_models(domain=ModelDomain.AML)

        # Get model entry
        entry = catalog.get("aml-transaction-risk-v1")

        # Search models
        matches = catalog.search("fraud detection")
        ```
    """

    def __init__(self) -> None:
        """Initialize the catalog."""
        self._models: dict[str, ModelEntry] = {}
        self._load_builtin_models()

    def _load_builtin_models(self) -> None:
        """Load built-in model entries."""
        # Transaction Risk Model
        self._models["aml-transaction-risk-v1"] = ModelEntry(
            model_id="aml-transaction-risk-v1",
            name="AML Transaction Risk Scorer",
            version="1.0.0",
            domain=ModelDomain.AML,
            architecture=ModelArchitecture.GRADIENT_BOOSTING,
            description=(
                "Gradient boosting model for transaction risk scoring. "
                "Trained on synthetic transaction data with structuring, "
                "velocity, and geographic risk features."
            ),
            metrics={
                "auc_roc": 0.92,
                "precision": 0.85,
                "recall": 0.78,
                "f1": 0.81,
            },
            input_features=[
                "amount",
                "transaction_type",
                "sender_country",
                "receiver_country",
                "hour_of_day",
                "day_of_week",
                "transaction_velocity_1h",
                "transaction_velocity_24h",
                "is_round_amount",
                "sender_account_age_days",
            ],
            output_schema={
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "risk_label": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            file_size_mb=25.5,
            tags=["aml", "transaction", "risk", "production-ready"],
        )

        # Anomaly Detector
        self._models["anomaly-detector-v1"] = ModelEntry(
            model_id="anomaly-detector-v1",
            name="Statistical Anomaly Detector",
            version="1.0.0",
            domain=ModelDomain.ANOMALY,
            architecture=ModelArchitecture.ENSEMBLE,
            description=(
                "Ensemble anomaly detector combining Isolation Forest, "
                "Local Outlier Factor, and autoencoder reconstruction."
            ),
            metrics={
                "auc_roc": 0.88,
                "precision_at_k": 0.72,
            },
            input_features=["*"],  # Any numeric features
            file_size_mb=15.2,
            tags=["anomaly", "unsupervised", "ensemble"],
        )

        # Entity Resolution
        self._models["entity-resolution-v1"] = ModelEntry(
            model_id="entity-resolution-v1",
            name="Entity Resolution Model",
            version="1.0.0",
            domain=ModelDomain.KYC,
            architecture=ModelArchitecture.NEURAL_NETWORK,
            description=(
                "Siamese neural network for entity matching and deduplication. "
                "Compares name, address, and document features."
            ),
            metrics={
                "precision": 0.94,
                "recall": 0.89,
                "f1": 0.91,
            },
            input_features=[
                "name",
                "address",
                "date_of_birth",
                "document_number",
            ],
            file_size_mb=42.0,
            tags=["kyc", "entity-resolution", "matching"],
        )

        # SAR Priority Model
        self._models["sar-priority-v1"] = ModelEntry(
            model_id="sar-priority-v1",
            name="SAR Priority Ranking Model",
            version="1.0.0",
            domain=ModelDomain.AML,
            architecture=ModelArchitecture.TRANSFORMER,
            description=(
                "Transformer model for prioritizing SAR investigations. "
                "Learns from historical filing decisions and outcomes."
            ),
            metrics={
                "auc_roc": 0.87,
                "ndcg_at_10": 0.82,
            },
            input_features=[
                "alert_features",
                "case_narrative",
                "transaction_summary",
            ],
            file_size_mb=180.0,
            tags=["aml", "sar", "nlp", "transformer"],
        )

        # Graph Risk Model
        self._models["graph-risk-v1"] = ModelEntry(
            model_id="graph-risk-v1",
            name="Transaction Graph Risk Model",
            version="1.0.0",
            domain=ModelDomain.AML,
            architecture=ModelArchitecture.GRAPH_NEURAL_NETWORK,
            description=(
                "Graph Neural Network for detecting suspicious patterns "
                "in transaction networks. Identifies shells, layering, "
                "and money mules."
            ),
            metrics={
                "auc_roc": 0.90,
                "precision": 0.82,
            },
            input_features=[
                "transaction_graph",
                "node_features",
                "edge_features",
            ],
            file_size_mb=95.0,
            tags=["aml", "graph", "network", "gnn"],
        )

    def get(self, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID.

        Args:
            model_id: Model identifier.

        Returns:
            ModelEntry or None.
        """
        return self._models.get(model_id)

    def list_models(
        self,
        *,
        domain: ModelDomain | None = None,
        architecture: ModelArchitecture | None = None,
        tag: str | None = None,
    ) -> list[ModelEntry]:
        """List available models.

        Args:
            domain: Filter by domain.
            architecture: Filter by architecture.
            tag: Filter by tag.

        Returns:
            List of matching model entries.
        """
        models = list(self._models.values())

        if domain:
            models = [m for m in models if m.domain == domain]

        if architecture:
            models = [m for m in models if m.architecture == architecture]

        if tag:
            models = [m for m in models if tag in m.tags]

        return models

    def search(self, query: str) -> list[ModelEntry]:
        """Search models by name or description.

        Args:
            query: Search query.

        Returns:
            Matching model entries.
        """
        query = query.lower()
        results = []

        for model in self._models.values():
            if (
                query in model.name.lower()
                or query in model.description.lower()
                or any(query in tag.lower() for tag in model.tags)
            ):
                results.append(model)

        return results

    def register(self, entry: ModelEntry) -> None:
        """Register a custom model entry.

        Args:
            entry: Model entry to register.
        """
        self._models[entry.model_id] = entry

    def __len__(self) -> int:
        """Number of models in catalog."""
        return len(self._models)
