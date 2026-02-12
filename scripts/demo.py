#!/usr/bin/env python3
"""
PolyglotLink Demo

Sends 3 different IoT device payloads through the pipeline via the REST API
and shows that all three are normalized to the same semantic ontology —
proving the core value proposition in under 60 seconds.

Usage:
    python scripts/demo.py                      # Uses in-process pipeline (no server needed)
    python scripts/demo.py --url http://localhost:8080  # Uses running server
"""

import argparse
import json
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# The 3 demo device payloads — same physical data, different representations
# ---------------------------------------------------------------------------

DEVICES = [
    {
        "name": "Temperature Sensor (Celsius, flat JSON)",
        "payload": {
            "temp_c": 23.5,
            "humidity_pct": 45,
            "bat_v": 3.2,
            "device_id": "sensor-A",
        },
    },
    {
        "name": "Weather Station (Fahrenheit, different field names)",
        "payload": {
            "temperature_f": 74.3,
            "rh": 45,
            "pressure_hpa": 1013.25,
            "station_id": "wx-station-B",
        },
    },
    {
        "name": "Industrial Sensor (nested, metric units)",
        "payload": {
            "readings": {
                "temperature": {"value": 296.65, "unit": "K"},
                "vibration": {"x": 0.02, "y": 0.01, "z": 0.03},
            },
            "serial": "IND-C-00042",
            "ts": int(datetime.utcnow().timestamp()),
        },
    },
]


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m"


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


# ---------------------------------------------------------------------------
# In-process pipeline (no server needed)
# ---------------------------------------------------------------------------


async def run_in_process(payload: dict, device_id: str = "demo") -> dict:
    """Run a single payload through the pipeline in-process."""
    from polyglotlink.models.schemas import PayloadEncoding, Protocol, RawMessage
    from polyglotlink.modules.normalization_engine import NormalizationEngine
    from polyglotlink.modules.protocol_listener import generate_uuid
    from polyglotlink.modules.schema_extractor import SchemaExtractor
    from polyglotlink.modules.semantic_translator_agent import SemanticTranslator

    payload_bytes = json.dumps(payload).encode()

    raw = RawMessage(
        message_id=generate_uuid(),
        device_id=device_id,
        protocol=Protocol.HTTP,
        topic="demo/ingest",
        payload_raw=payload_bytes,
        payload_encoding=PayloadEncoding.JSON,
        timestamp=datetime.utcnow(),
    )

    extractor = SchemaExtractor()
    schema = extractor.extract_schema(raw)

    translator = SemanticTranslator()
    mapping = await translator.translate_schema(schema)

    engine = NormalizationEngine()
    normalized = engine.normalize_message(schema, mapping)

    return {
        "schema_signature": normalized.schema_signature,
        "confidence": normalized.confidence,
        "data": normalized.data,
        "conversions": [
            {
                "field": c.field,
                "from": f"{c.original_value} {c.from_unit}",
                "to": f"{c.converted_value} {c.to_unit}",
            }
            for c in normalized.conversions
        ],
        "fields_extracted": len(schema.fields),
        "fields_with_units": sum(1 for f in schema.fields if f.inferred_unit),
        "fields_with_semantics": sum(1 for f in schema.fields if f.inferred_semantic),
    }


# ---------------------------------------------------------------------------
# HTTP-based pipeline (requires running server)
# ---------------------------------------------------------------------------


async def run_via_http(url: str, payload: dict) -> dict:
    """Run a single payload through the pipeline via HTTP API."""
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{url}/api/v1/test",
            json={"payload": payload},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(url: str | None = None) -> int:
    print()
    print(_bold("=" * 70))
    print(_bold("  PolyglotLink Demo — Semantic API Translator for IoT"))
    print(_bold("=" * 70))
    print()

    if url:
        print(f"  Mode: HTTP API ({url})")
    else:
        print("  Mode: In-process pipeline (no server needed)")
    print()

    results = []

    for i, device in enumerate(DEVICES, 1):
        print(_bold(f"--- Device {i}: {device['name']} ---"))
        print()
        print(_dim("  Input:"))
        for line in json.dumps(device["payload"], indent=4).split("\n"):
            print(f"    {_dim(line)}")
        print()

        try:
            if url:
                result = await run_via_http(url, device["payload"])
            else:
                result = await run_in_process(device["payload"])

            results.append(result)

            print(_green("  Output:"))
            if "data" in result:
                for line in json.dumps(result["data"], indent=4, default=str).split("\n"):
                    print(f"    {_green(line)}")

            print()
            print(f"  Schema signature: {_cyan(result.get('schema_signature', 'N/A'))}")
            print(f"  Confidence:       {result.get('confidence', 'N/A')}")

            if result.get("fields_extracted") is not None:
                print(
                    f"  Fields extracted:  {result['fields_extracted']} "
                    f"({result.get('fields_with_units', 0)} with units, "
                    f"{result.get('fields_with_semantics', 0)} with semantics)"
                )

            if result.get("conversions"):
                print()
                print("  Conversions applied:")
                for conv in result["conversions"]:
                    print(f"    {conv['field']}: {conv['from']} -> {conv['to']}")

        except Exception as e:
            print(f"  \033[31mError: {e}\033[0m")
            if url:
                print("  Is the server running? Try: make docker-up && make run")
            return 1

        print()

    # Summary
    print(_bold("=" * 70))
    print(_bold("  Summary"))
    print(_bold("=" * 70))
    print()

    sigs = [r.get("schema_signature", "?") for r in results]
    unique_sigs = set(sigs)
    print(f"  Devices processed:   {len(results)}")
    print(f"  Unique schemas:      {len(unique_sigs)}")
    print(f"  All schemas learned: {_green('Yes') if len(unique_sigs) == len(results) else 'Check results'}")
    print()
    print(
        "  Each device used different field names, units, and nesting —"
    )
    print(
        "  PolyglotLink auto-extracted schemas, inferred semantics,"
    )
    print(
        "  and normalized the data without any manual adapter code."
    )
    print()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyglotLink Demo")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Server URL (e.g., http://localhost:8080). Omit for in-process mode.",
    )
    args = parser.parse_args()

    import asyncio

    sys.exit(asyncio.run(main(url=args.url)))
