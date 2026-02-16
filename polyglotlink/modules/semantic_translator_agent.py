"""
PolyglotLink Semantic Translator Agent Module

This module uses LLMs and embedding-based resolution to map extracted
IoT fields to standardized ontology concepts.
"""

import json
import re
from datetime import datetime, timezone

import structlog

from polyglotlink.models.schemas import (
    CachedMapping,
    ExtractedField,
    ExtractedSchema,
    FieldMapping,
    ResolutionMethod,
    SemanticMapping,
    SemanticTranslatorConfig,
    SuggestedConcept,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# LLM Translation Prompt
# ============================================================================

TRANSLATION_PROMPT_TEMPLATE = """
You are an IoT data semantics expert. Your task is to map raw device fields to standardized ontology concepts.

## Device Context
- Protocol: {protocol}
- Topic/Path: {topic}
- Device ID: {device_id}

## Raw Fields
{fields_table}

## Available Ontology Concepts
{ontology_concepts}

## Task
For each raw field, determine:
1. The standardized concept it maps to (from ontology or suggest new)
2. The standardized field name
3. Unit conversion if needed (source unit → target unit)
4. Confidence score (0.0-1.0)

## Output Format (JSON only)
{{
  "mappings": [
    {{
      "source_field": "original field name",
      "target_concept": "ontology concept ID",
      "target_field": "standardized field name",
      "source_unit": "detected unit or null",
      "target_unit": "standard unit",
      "conversion_formula": "formula if conversion needed, else null",
      "confidence": 0.95,
      "reasoning": "brief explanation"
    }}
  ],
  "device_context": "inferred device type/purpose",
  "suggested_new_concepts": [
    {{
      "concept_id": "suggested_concept_name",
      "description": "what this concept represents",
      "unit": "standard unit",
      "datatype": "float|integer|string|boolean"
    }}
  ]
}}

Generate the mapping:
"""


# ============================================================================
# Ontology Concept Definitions
# ============================================================================

# Default ontology concepts for when no external ontology is available
DEFAULT_ONTOLOGY_CONCEPTS = [
    {
        "concept_id": "temperature_celsius",
        "description": "Temperature in degrees Celsius",
        "unit": "celsius",
        "datatype": "float",
        "aliases": ["tmp", "temp", "temperature", "t", "thermo"],
    },
    {
        "concept_id": "humidity_percent",
        "description": "Relative humidity percentage",
        "unit": "percent",
        "datatype": "float",
        "aliases": ["hum", "humidity", "rh", "relative_humidity"],
    },
    {
        "concept_id": "pressure_pascal",
        "description": "Atmospheric or process pressure",
        "unit": "pascal",
        "datatype": "float",
        "aliases": ["press", "pressure", "baro", "atm"],
    },
    {
        "concept_id": "co2_ppm",
        "description": "Carbon dioxide concentration",
        "unit": "ppm",
        "datatype": "float",
        "aliases": ["co2", "carbon_dioxide", "co2_level"],
    },
    {
        "concept_id": "voltage_volt",
        "description": "Electrical voltage",
        "unit": "volt",
        "datatype": "float",
        "aliases": ["volt", "voltage", "v", "vdc", "vac"],
    },
    {
        "concept_id": "current_ampere",
        "description": "Electrical current",
        "unit": "ampere",
        "datatype": "float",
        "aliases": ["current", "amp", "ampere", "i", "amps"],
    },
    {
        "concept_id": "power_watt",
        "description": "Electrical power",
        "unit": "watt",
        "datatype": "float",
        "aliases": ["power", "watt", "w", "watts"],
    },
    {
        "concept_id": "latitude_degrees",
        "description": "Geographic latitude",
        "unit": "degrees",
        "datatype": "float",
        "aliases": ["lat", "latitude"],
    },
    {
        "concept_id": "longitude_degrees",
        "description": "Geographic longitude",
        "unit": "degrees",
        "datatype": "float",
        "aliases": ["lon", "lng", "longitude"],
    },
    {
        "concept_id": "speed_mps",
        "description": "Speed in meters per second",
        "unit": "meters_per_second",
        "datatype": "float",
        "aliases": ["speed", "velocity", "spd"],
    },
    {
        "concept_id": "battery_percent",
        "description": "Battery charge level",
        "unit": "percent",
        "datatype": "float",
        "aliases": ["battery", "batt", "bat_level", "charge"],
    },
    {
        "concept_id": "signal_rssi",
        "description": "Signal strength (RSSI)",
        "unit": "dbm",
        "datatype": "integer",
        "aliases": ["rssi", "signal", "signal_strength"],
    },
    {
        "concept_id": "connectivity_status",
        "description": "Online/offline status",
        "unit": "boolean",
        "datatype": "boolean",
        "aliases": ["online", "connected", "alive", "status"],
    },
]


def build_fields_table(fields: list[ExtractedField]) -> str:
    """Build a markdown table of fields for the LLM prompt."""
    rows = ["| Field | Value | Type | Inferred Unit | Semantic Hint |"]
    rows.append("|-------|-------|------|---------------|---------------|")

    for field in fields:
        value_str = str(field.value)
        if len(value_str) > 30:
            value_str = value_str[:27] + "..."

        rows.append(
            f"| {field.key} | {value_str} | "
            f"{field.value_type} | {field.inferred_unit or '-'} | "
            f"{field.inferred_semantic or '-'} |"
        )

    return "\n".join(rows)


def build_ontology_context(
    fields: list[ExtractedField], ontology_concepts: list[dict] | None = None
) -> str:
    """Build ontology context for the LLM prompt."""
    concepts = ontology_concepts or DEFAULT_ONTOLOGY_CONCEPTS

    # Filter to relevant concepts based on field hints
    relevant_concepts = []
    seen_ids = set()

    for field in fields:
        if field.inferred_semantic:
            for concept in concepts:
                if concept["concept_id"] not in seen_ids:
                    # Check if any alias matches the semantic hint
                    aliases = concept.get("aliases", [])
                    if any(alias in field.inferred_semantic.lower() for alias in aliases):
                        relevant_concepts.append(concept)
                        seen_ids.add(concept["concept_id"])

    # Add popular concepts that weren't matched
    for concept in concepts[:20]:
        if concept["concept_id"] not in seen_ids:
            relevant_concepts.append(concept)
            seen_ids.add(concept["concept_id"])

    # Format for prompt
    lines = []
    for concept in relevant_concepts[:50]:
        aliases = ", ".join(concept.get("aliases", [])[:3])
        lines.append(
            f"- {concept['concept_id']}: {concept['description']} "
            f"(unit: {concept['unit']}, type: {concept['datatype']}, aliases: {aliases})"
        )

    return "\n".join(lines)


# ============================================================================
# Embedding-Based Resolution
# ============================================================================


class EmbeddingResolver:
    """Resolves field mappings using vector similarity."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        threshold: float = 0.85,
        weaviate_client=None,
        openai_client=None,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.weaviate = weaviate_client
        self.openai = openai_client
        self._concept_embeddings: dict[str, list[float]] = {}
        self._initialized = False

    async def initialize(self, concepts: list[dict]) -> None:
        """Pre-compute embeddings for all ontology concepts."""
        if self._initialized:
            return

        for concept in concepts:
            # Create embedding text from concept info
            text = f"{concept['concept_id']} {concept['description']} " + " ".join(
                concept.get("aliases", [])
            )

            embedding = await self._get_embedding(text)
            if embedding:
                self._concept_embeddings[concept["concept_id"]] = embedding

        self._initialized = True
        logger.info("Embedding resolver initialized", concepts=len(self._concept_embeddings))

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using OpenAI or fallback."""
        if self.openai:
            try:
                response = await self.openai.embeddings.create(
                    model=self.embedding_model, input=text
                )
                if response.data and len(response.data) > 0:
                    return response.data[0].embedding
                logger.warning("OpenAI embedding returned empty response")
                return None
            except Exception as e:
                logger.warning("OpenAI embedding failed", error=str(e))

        # Fallback: token-overlap pseudo-embedding for when OpenAI is unavailable.
        # Uses a bag-of-words approach so that semantically related inputs (sharing
        # common tokens like "temperature", "celsius") produce similar vectors, unlike
        # a cryptographic hash which destroys similarity.
        tokens = set(re.split(r"[\s_\-./]+", text.lower()))
        # Build a stable vocabulary from the known ontology aliases
        vocab: list[str] = []
        for concept in DEFAULT_ONTOLOGY_CONCEPTS:
            for alias in concept.get("aliases", []):
                if alias not in vocab:
                    vocab.append(alias)
            cid = concept["concept_id"]
            if cid not in vocab:
                vocab.append(cid)
        # Create a sparse vector based on token membership
        import hashlib as _hl

        dim = max(len(vocab), 32)
        vec = [0.0] * dim
        for token in tokens:
            if token in vocab:
                vec[vocab.index(token)] = 1.0
            else:
                # Hash unknown tokens to a stable bucket
                idx = int(_hl.md5(token.encode()).hexdigest(), 16) % dim  # nosec B324
                vec[idx] = 0.3
        # Normalize to unit vector
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Vectors may differ in length when switching between embedding
        backends (e.g. OpenAI vs. local fallback).  We use the shorter
        length so that a dimension mismatch degrades gracefully instead
        of raising a ``ValueError``.
        """
        length = min(len(a), len(b))
        if length == 0:
            return 0.0
        dot_product = sum(a[i] * b[i] for i in range(length))
        norm_a = sum(a[i] * a[i] for i in range(length)) ** 0.5
        norm_b = sum(b[i] * b[i] for i in range(length)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def resolve(
        self, field: ExtractedField, concepts: list[dict] | None = None
    ) -> FieldMapping | None:
        """
        Attempt to resolve field mapping using vector similarity.
        """
        if not self._initialized:
            await self.initialize(concepts or DEFAULT_ONTOLOGY_CONCEPTS)

        # Generate embedding for field
        query_text = f"{field.key} {field.inferred_semantic or ''} {field.inferred_unit or ''}"
        query_embedding = await self._get_embedding(query_text)

        if not query_embedding:
            return None

        # Check Weaviate if available
        if self.weaviate:
            try:
                results = (
                    self.weaviate.query.get(
                        "OntologyConcept",
                        ["concept_id", "canonical_name", "unit", "datatype", "aliases"],
                    )
                    .with_near_vector({"vector": query_embedding, "certainty": self.threshold})
                    .with_limit(5)
                    .do()
                )

                if results.get("data", {}).get("Get", {}).get("OntologyConcept"):
                    best_match = results["data"]["Get"]["OntologyConcept"][0]
                    certainty = best_match.get("_additional", {}).get("certainty", 0)

                    if certainty >= self.threshold:
                        return FieldMapping(
                            source_field=field.key,
                            target_concept=best_match["concept_id"],
                            target_field=best_match["canonical_name"],
                            source_unit=field.inferred_unit,
                            target_unit=best_match["unit"],
                            conversion_formula=None,  # Determined later
                            confidence=certainty,
                            resolution_method=ResolutionMethod.EMBEDDING,
                        )
            except Exception as e:
                logger.warning("Weaviate query failed", error=str(e))

        # Fallback: local embedding search
        best_match = None
        best_similarity = 0.0

        for concept_id, embedding in self._concept_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = concept_id

        if best_match and best_similarity >= self.threshold:
            # Find the concept details
            concepts_list = concepts or DEFAULT_ONTOLOGY_CONCEPTS
            concept = next((c for c in concepts_list if c["concept_id"] == best_match), None)

            if concept:
                return FieldMapping(
                    source_field=field.key,
                    target_concept=best_match,
                    target_field=best_match,
                    source_unit=field.inferred_unit,
                    target_unit=concept["unit"],
                    conversion_formula=None,
                    confidence=best_similarity,
                    resolution_method=ResolutionMethod.EMBEDDING,
                )

        return None


# ============================================================================
# LLM-Based Translation
# ============================================================================


class LLMTranslator:
    """Handles LLM-based semantic translation."""

    def __init__(self, config: SemanticTranslatorConfig, openai_client=None):
        self.config = config
        self.openai = openai_client

    async def translate(
        self,
        schema: ExtractedSchema,
        fields: list[ExtractedField],
        ontology_concepts: list[dict] | None = None,
    ) -> tuple[list[FieldMapping], str | None, list[SuggestedConcept]]:
        """
        Use LLM to translate fields to ontology concepts.
        Returns: (mappings, device_context, suggested_concepts)
        """
        # Build prompt
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(
            protocol=schema.protocol.value,
            topic=schema.topic,
            device_id=schema.device_id,
            fields_table=build_fields_table(fields),
            ontology_concepts=build_ontology_context(fields, ontology_concepts),
        )

        # Call LLM (pass fields for rule-based fallback when LLM is unavailable)
        response_text = await self._call_llm(prompt, fields=fields)

        if not response_text:
            return [], None, []

        # Parse response
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                try:
                    response = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM response as JSON")
                    return [], None, []
            else:
                logger.error("No JSON found in LLM response")
                return [], None, []

        # Convert to FieldMapping objects
        mappings = []
        for m in response.get("mappings", []):
            try:
                mapping = FieldMapping(
                    source_field=m["source_field"],
                    target_concept=m["target_concept"],
                    target_field=m["target_field"],
                    source_unit=m.get("source_unit"),
                    target_unit=m.get("target_unit"),
                    conversion_formula=m.get("conversion_formula"),
                    confidence=float(m.get("confidence", 0.8)),
                    resolution_method=ResolutionMethod.LLM,
                    reasoning=m.get("reasoning"),
                )
                mappings.append(mapping)
            except Exception as e:
                logger.warning("Failed to parse mapping", error=str(e), mapping=m)

        # Parse suggested concepts
        suggested_concepts = []
        for sc in response.get("suggested_new_concepts", []):
            try:
                concept = SuggestedConcept(
                    concept_id=sc["concept_id"],
                    description=sc["description"],
                    unit=sc["unit"],
                    datatype=sc["datatype"],
                    aliases=sc.get("aliases", []),
                )
                suggested_concepts.append(concept)
            except Exception as e:
                logger.warning("Failed to parse suggested concept", error=str(e))

        device_context = response.get("device_context")

        return mappings, device_context, suggested_concepts

    async def _call_llm(
        self, prompt: str, fields: list[ExtractedField] | None = None
    ) -> str | None:
        """Call the configured LLM."""
        if self.openai:
            for attempt in range(self.config.max_llm_retries):
                try:
                    response = await self.openai.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an IoT data semantics expert. Respond only with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens,
                    )
                    if response.choices and len(response.choices) > 0:
                        return response.choices[0].message.content
                    logger.warning("LLM returned empty choices")
                    return None
                except Exception as e:
                    logger.warning("LLM call failed", attempt=attempt + 1, error=str(e))
                    if attempt == self.config.max_llm_retries - 1:
                        raise

        # Fallback: rule-based translation using field hints
        return self._rule_based_fallback(prompt, fields=fields)

    def _rule_based_fallback(self, _prompt: str, fields: list[ExtractedField] | None = None) -> str:
        """Rule-based fallback when LLM is unavailable.

        Uses inferred semantic hints and unit patterns from the schema extractor
        to produce reasonable mappings without an LLM.
        """
        if not fields:
            return json.dumps(
                {"mappings": [], "device_context": "unknown", "suggested_new_concepts": []}
            )

        # Build a lookup from semantic hint to ontology concept
        hint_to_concept: dict[str, dict] = {}
        for concept in DEFAULT_ONTOLOGY_CONCEPTS:
            for alias in concept.get("aliases", []):
                hint_to_concept[alias] = concept

        mappings = []
        for field in fields:
            matched_concept = None

            # Try matching via semantic hint
            if field.inferred_semantic:
                for alias, concept in hint_to_concept.items():
                    if alias in field.inferred_semantic.lower() or field.inferred_semantic.lower() in alias:
                        matched_concept = concept
                        break

            # Try matching via field key directly
            if not matched_concept:
                key_lower = field.key.lower().replace(".", "_")
                for alias, concept in hint_to_concept.items():
                    if alias in key_lower:
                        matched_concept = concept
                        break

            if matched_concept:
                mappings.append({
                    "source_field": field.key,
                    "target_concept": matched_concept["concept_id"],
                    "target_field": matched_concept["concept_id"],
                    "source_unit": field.inferred_unit,
                    "target_unit": matched_concept["unit"],
                    "conversion_formula": None,
                    "confidence": 0.7,
                    "reasoning": "rule-based fallback (no LLM available)",
                })

        return json.dumps(
            {"mappings": mappings, "device_context": "unknown", "suggested_new_concepts": []}
        )


# ============================================================================
# Semantic Translator
# ============================================================================


class SemanticTranslator:
    """
    Main semantic translation class.
    Uses cache, embeddings, then LLM fallback.
    """

    def __init__(
        self,
        config: SemanticTranslatorConfig | None = None,
        openai_client=None,
        weaviate_client=None,
        ontology_registry=None,
    ):
        self.config = config or SemanticTranslatorConfig()
        self.openai = openai_client
        self.ontology_registry = ontology_registry

        self.embedding_resolver = EmbeddingResolver(
            embedding_model=self.config.embedding_model,
            threshold=self.config.embedding_threshold,
            weaviate_client=weaviate_client,
            openai_client=openai_client,
        )

        self.llm_translator = LLMTranslator(config=self.config, openai_client=openai_client)

    async def translate_schema(self, schema: ExtractedSchema) -> SemanticMapping:
        """
        Main translation function.
        Uses cache, embeddings, then LLM fallback.
        """
        # Check cache first
        if schema.cached_mapping:
            logger.info("Using cached mapping", signature=schema.schema_signature)
            return self._apply_cached_mapping(schema, schema.cached_mapping)

        mappings: list[FieldMapping] = []
        fields_needing_llm: list[ExtractedField] = []

        # Try embedding resolution for each field
        for field in schema.fields:
            # Skip timestamps and identifiers
            if field.is_timestamp or field.is_identifier:
                mappings.append(self._create_passthrough_mapping(field))
                continue

            embedding_result = await self.embedding_resolver.resolve(field)

            if (
                embedding_result
                and embedding_result.confidence >= self.config.min_confidence_threshold
            ):
                mappings.append(embedding_result)
            else:
                fields_needing_llm.append(field)

        # Call LLM for unresolved fields
        device_context = None
        if fields_needing_llm:
            llm_mappings, device_context, suggested_concepts = await self.llm_translator.translate(
                schema, fields_needing_llm
            )

            mappings.extend(llm_mappings)

            # Learn new concepts if suggested and enabled
            if self.config.enable_concept_learning and suggested_concepts:
                for concept in suggested_concepts:
                    self._learn_concept(concept)

        # Validate all mappings
        validated_mappings = []
        for mapping in mappings:
            if self._validate_mapping(mapping):
                validated_mappings.append(mapping)
            else:
                logger.warning("Invalid mapping rejected", source_field=mapping.source_field)
                validated_mappings.append(self._create_fallback_mapping(mapping.source_field))

        # Compute overall confidence
        overall_confidence = self._compute_overall_confidence(validated_mappings)

        result = SemanticMapping(
            message_id=schema.message_id,
            device_id=schema.device_id,
            schema_signature=schema.schema_signature,
            field_mappings=validated_mappings,
            device_context=device_context,
            confidence=overall_confidence,
            llm_generated=len(fields_needing_llm) > 0,
            translated_at=datetime.now(timezone.utc),
        )

        logger.info(
            "Schema translated",
            message_id=schema.message_id,
            mappings=len(validated_mappings),
            llm_used=result.llm_generated,
            confidence=overall_confidence,
        )

        return result

    def _apply_cached_mapping(
        self, schema: ExtractedSchema, cached: CachedMapping
    ) -> SemanticMapping:
        """Apply a cached mapping to a schema."""
        return SemanticMapping(
            message_id=schema.message_id,
            device_id=schema.device_id,
            schema_signature=schema.schema_signature,
            field_mappings=cached.field_mappings,
            device_context=None,
            confidence=cached.confidence,
            llm_generated=False,
            translated_at=datetime.now(timezone.utc),
        )

    def _create_passthrough_mapping(self, field: ExtractedField) -> FieldMapping:
        """Create a passthrough mapping for timestamps/identifiers."""
        return FieldMapping(
            source_field=field.key,
            target_concept=f"_{field.inferred_semantic or 'passthrough'}",
            target_field=field.key,
            source_unit=field.inferred_unit,
            target_unit=field.inferred_unit,
            conversion_formula=None,
            confidence=1.0,
            resolution_method=ResolutionMethod.PASSTHROUGH,
        )

    def _create_fallback_mapping(self, source_field: str) -> FieldMapping:
        """Create a fallback mapping for unresolved fields."""
        return FieldMapping(
            source_field=source_field,
            target_concept="_unmapped",
            target_field=f"_unmapped.{source_field}",
            source_unit=None,
            target_unit=None,
            conversion_formula=None,
            confidence=0.0,
            resolution_method=ResolutionMethod.PASSTHROUGH,
        )

    def _validate_mapping(self, mapping: FieldMapping) -> bool:
        """Validate a field mapping against ontology constraints."""
        # Basic validation
        if not mapping.target_concept or not mapping.target_field:
            return False

        if mapping.confidence < 0 or mapping.confidence > 1:
            return False

        # Ontology validation if available
        if self.ontology_registry:
            concept = self.ontology_registry.get_concept(mapping.target_concept)
            if concept is None and not mapping.target_concept.startswith("_"):
                logger.warning("Unknown target concept", concept=mapping.target_concept)
                # Allow it but log warning

        return True

    def _compute_overall_confidence(self, mappings: list[FieldMapping]) -> float:
        """Compute overall confidence from individual mappings."""
        if not mappings:
            return 0.0

        # Weighted average, excluding passthrough mappings
        weighted_sum = 0.0
        weight_total = 0.0

        for mapping in mappings:
            if mapping.resolution_method != ResolutionMethod.PASSTHROUGH:
                weight = 1.0
                if mapping.resolution_method == ResolutionMethod.LLM:
                    weight = 0.9  # Slightly lower weight for LLM
                elif mapping.resolution_method == ResolutionMethod.CACHE:
                    weight = 1.1  # Slightly higher weight for cached

                weighted_sum += mapping.confidence * weight
                weight_total += weight

        if weight_total == 0:
            return 1.0  # All passthrough

        return min(1.0, weighted_sum / weight_total)

    def _learn_concept(self, concept: SuggestedConcept) -> None:
        """Add a new concept to the ontology."""
        if self.ontology_registry:
            try:
                self.ontology_registry.add_concept(concept)
                logger.info("Learned new concept", concept_id=concept.concept_id)
            except Exception as e:
                logger.warning(
                    "Failed to learn concept", concept_id=concept.concept_id, error=str(e)
                )
