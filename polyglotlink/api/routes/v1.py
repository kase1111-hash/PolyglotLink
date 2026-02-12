"""
PolyglotLink REST API v1

User-facing endpoints for ingesting payloads, querying schemas,
and testing the pipeline.
"""

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Request body for POST /ingest and POST /test."""

    payload: dict[str, Any] = Field(..., description="Raw device payload (JSON)")
    device_id: str = Field(default="api-device", description="Device identifier")
    protocol: str = Field(default="HTTP", description="Source protocol hint")
    topic: str = Field(default="api/ingest", description="Simulated topic/path")


class IngestResponse(BaseModel):
    """Response from the ingest/test endpoints."""

    message_id: str
    device_id: str
    timestamp: str
    data: dict[str, Any]
    metadata: dict[str, Any]
    context: str | None
    schema_signature: str
    confidence: float
    conversions: list[dict[str, Any]]
    validation_errors: list[dict[str, Any]]
    published: bool


class SchemaListItem(BaseModel):
    """Summary of a cached schema."""

    schema_signature: str
    field_count: int
    confidence: float
    source: str
    created_at: str
    hits: int


class HealthDetail(BaseModel):
    """Detailed health response."""

    status: str
    running: bool
    messages_received: int
    messages_processed: int
    messages_failed: int
    uptime_seconds: float | None
    cache_schemas: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_server(request: Request):
    """Get the PolyglotLinkServer from app state, or raise 503."""
    server = request.app.state.server
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return server


async def _run_pipeline(server, body: IngestRequest, publish: bool) -> IngestResponse:
    """Run a payload through the full pipeline."""
    from polyglotlink.models.schemas import PayloadEncoding, Protocol, RawMessage
    from polyglotlink.modules.protocol_listener import generate_uuid

    payload_bytes = json.dumps(body.payload).encode()

    raw = RawMessage(
        message_id=generate_uuid(),
        device_id=body.device_id,
        protocol=Protocol[body.protocol.upper()],
        topic=body.topic,
        payload_raw=payload_bytes,
        payload_encoding=PayloadEncoding.JSON,
        timestamp=datetime.utcnow(),
    )

    # Step 1: Extract schema
    schema = server._schema_extractor.extract_schema(raw)

    # Step 2: Translate to semantic mapping
    mapping = await server._semantic_translator.translate_schema(schema)

    # Step 3: Normalize values
    normalized = server._normalization_engine.normalize_message(schema, mapping)

    # Step 4: Optionally publish
    published = False
    if publish and server._output_broker:
        await server._output_broker.publish(normalized)
        published = True

    return IngestResponse(
        message_id=normalized.message_id,
        device_id=normalized.device_id,
        timestamp=normalized.timestamp.isoformat(),
        data=normalized.data,
        metadata=normalized.metadata,
        context=normalized.context,
        schema_signature=normalized.schema_signature,
        confidence=normalized.confidence,
        conversions=[c.model_dump() for c in normalized.conversions],
        validation_errors=[e.model_dump() for e in normalized.validation_errors],
        published=published,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthDetail)
async def health_detail(request: Request):
    """Detailed health check with pipeline status and cache stats."""
    server = request.app.state.server

    if server is None:
        return HealthDetail(
            status="starting",
            running=False,
            messages_received=0,
            messages_processed=0,
            messages_failed=0,
            uptime_seconds=None,
            cache_schemas=0,
        )

    metrics = server.get_metrics()
    cache_count = 0
    if server._schema_extractor and server._schema_extractor.cache:
        cache_count = len(server._schema_extractor.cache.list_all())

    return HealthDetail(
        status="healthy" if server._running else "stopped",
        running=server._running,
        messages_received=metrics.get("messages_received", 0),
        messages_processed=metrics.get("messages_processed", 0),
        messages_failed=metrics.get("messages_failed", 0),
        uptime_seconds=metrics.get("uptime_seconds"),
        cache_schemas=cache_count,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest, publish: bool = Query(default=True)):
    """Ingest a raw payload through the full pipeline.

    Set ?publish=false to process without sending to output brokers.
    """
    server = _get_server(request)

    try:
        return await _run_pipeline(server, body, publish=publish)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid protocol: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/test", response_model=IngestResponse)
async def test_pipeline(request: Request, body: IngestRequest):
    """Dry-run: process a payload without publishing to output brokers.

    Identical to POST /ingest?publish=false — a convenience endpoint for testing.
    """
    server = _get_server(request)

    try:
        return await _run_pipeline(server, body, publish=False)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid protocol: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/schemas", response_model=list[SchemaListItem])
async def list_schemas(request: Request):
    """List all cached schema signatures with metadata."""
    server = _get_server(request)

    if not server._schema_extractor or not server._schema_extractor.cache:
        return []

    items = server._schema_extractor.cache.list_all()
    return [SchemaListItem(**item) for item in items]


@router.get("/schemas/{signature}")
async def get_schema(signature: str, request: Request):
    """Get a specific cached schema mapping by signature."""
    server = _get_server(request)

    if not server._schema_extractor or not server._schema_extractor.cache:
        raise HTTPException(status_code=404, detail="Schema cache not available")

    mapping = server._schema_extractor.cache.get(signature)
    if mapping is None:
        raise HTTPException(status_code=404, detail=f"Schema '{signature}' not found")

    return mapping.model_dump(mode="json")
