# PolyglotLink Architecture

This document describes the system architecture, component interactions, and data flow of PolyglotLink.

## System Overview

PolyglotLink is a semantic API translator for IoT device ecosystems. It provides a unified semantic layer that automatically translates heterogeneous device protocols and data formats into a normalized, semantically-enriched message format.

```mermaid
graph TB
    subgraph "IoT Devices"
        D1[Environmental Sensors]
        D2[Power Meters]
        D3[GPS Trackers]
        D4[Industrial PLCs]
        D5[Smart Home Devices]
    end

    subgraph "Protocol Layer"
        MQTT[MQTT Broker]
        COAP[CoAP Server]
        HTTP[HTTP/REST]
        WS[WebSocket]
        MODBUS[Modbus Gateway]
        OPCUA[OPC-UA Server]
    end

    subgraph "PolyglotLink Core"
        PL[Protocol Listener]
        SE[Schema Extractor]
        ST[Semantic Translator]
        NE[Normalization Engine]
        OB[Output Broker]
    end

    subgraph "Storage & Services"
        Redis[(Redis Cache)]
        Neo4j[(Neo4j Ontology)]
        Weaviate[(Weaviate Vectors)]
        TimescaleDB[(TimescaleDB)]
    end

    subgraph "Outputs"
        Kafka[Apache Kafka]
        MQTT_OUT[MQTT Republish]
        HTTP_OUT[HTTP Webhook]
        WS_OUT[WebSocket Broadcast]
    end

    D1 & D2 & D3 & D4 & D5 --> MQTT & COAP & HTTP & WS & MODBUS & OPCUA
    MQTT & COAP & HTTP & WS & MODBUS & OPCUA --> PL
    PL --> SE
    SE --> ST
    ST --> NE
    NE --> OB

    SE <--> Redis
    ST <--> Neo4j
    ST <--> Weaviate
    NE <--> TimescaleDB

    OB --> Kafka & MQTT_OUT & HTTP_OUT & WS_OUT
```

## Component Architecture

### Core Components

```mermaid
graph LR
    subgraph "Protocol Listener"
        MH[MQTT Handler]
        CH[CoAP Handler]
        HH[HTTP Handler]
        WSH[WebSocket Handler]
        MBH[Modbus Handler]
        OUH[OPC-UA Handler]
    end

    subgraph "Schema Extractor"
        ED[Encoding Detector]
        JP[JSON Parser]
        XP[XML Parser]
        CP[CBOR Parser]
        TI[Type Inferrer]
        UI[Unit Inferrer]
        SC[Schema Cache]
    end

    subgraph "Semantic Translator"
        EC[Embedding Client]
        LC[LLM Client]
        OR[Ontology Resolver]
        MC[Mapping Cache]
    end

    subgraph "Normalization Engine"
        UC[Unit Converter]
        TC[Type Coercer]
        VV[Value Validator]
        ME[Metadata Enricher]
    end

    subgraph "Output Broker"
        KP[Kafka Producer]
        MP[MQTT Publisher]
        HP[HTTP Client]
        WP[WebSocket Publisher]
        TP[TimescaleDB Writer]
    end
```

## Data Flow

### Message Processing Pipeline

```mermaid
sequenceDiagram
    participant Device
    participant PL as Protocol Listener
    participant SE as Schema Extractor
    participant ST as Semantic Translator
    participant NE as Normalization Engine
    participant OB as Output Broker
    participant Cache as Redis Cache
    participant LLM as LLM/Embedding API

    Device->>PL: Raw Message (MQTT/CoAP/HTTP)
    PL->>PL: Detect Encoding
    PL->>PL: Extract Device ID
    PL->>SE: RawMessage

    SE->>SE: Parse Payload
    SE->>SE: Flatten Nested Fields
    SE->>SE: Infer Types & Units
    SE->>Cache: Check Schema Cache

    alt Schema Cached
        Cache-->>SE: Cached Schema
    else Schema Not Cached
        SE->>SE: Generate Schema Signature
        SE->>Cache: Store Schema
    end

    SE->>ST: ExtractedSchema

    ST->>Cache: Check Mapping Cache

    alt Mapping Cached
        Cache-->>ST: Cached Mapping
    else Mapping Not Cached
        ST->>LLM: Generate Embeddings
        LLM-->>ST: Field Embeddings
        ST->>ST: Match to Ontology

        alt Low Confidence Match
            ST->>LLM: LLM Translation
            LLM-->>ST: Field Mappings
        end

        ST->>Cache: Store Mapping
    end

    ST->>NE: SemanticMapping

    NE->>NE: Apply Unit Conversions
    NE->>NE: Coerce Types
    NE->>NE: Validate Values
    NE->>NE: Enrich Metadata
    NE->>OB: NormalizedMessage

    OB->>OB: Format Output (JSON/JSON-LD)

    par Parallel Publishing
        OB->>Kafka: Publish
        OB->>MQTT: Republish
        OB->>HTTP: POST Webhook
        OB->>TimescaleDB: Insert
    end
```

### Schema Detection Flow

```mermaid
flowchart TD
    A[Receive Payload] --> B{Detect Encoding}
    B -->|JSON| C[Parse JSON]
    B -->|XML| D[Parse XML]
    B -->|CBOR| E[Parse CBOR]
    B -->|Binary| F[Binary Handler]

    C & D & E --> G[Flatten Nested Structure]
    F --> H[Protocol-Specific Parser]

    G --> I[Extract Fields]
    H --> I

    I --> J[Infer Value Types]
    J --> K[Infer Units from Names]
    K --> L[Detect Semantic Hints]
    L --> M[Generate Schema Signature]

    M --> N{Schema in Cache?}
    N -->|Yes| O[Return Cached Schema]
    N -->|No| P[Create New Schema]
    P --> Q[Cache Schema]
    Q --> R[Return Schema]
    O --> R
```

### Semantic Translation Flow

```mermaid
flowchart TD
    A[Receive Schema] --> B{Mapping Cached?}

    B -->|Yes| C[Return Cached Mapping]

    B -->|No| D[Load Ontology Concepts]
    D --> E[Generate Field Embeddings]
    E --> F[Calculate Similarity Scores]

    F --> G{High Confidence Match?}

    G -->|Yes, >0.85| H[Direct Mapping]
    G -->|Medium, 0.7-0.85| I[Embedding-Based Mapping]
    G -->|Low, <0.7| J[LLM Translation]

    H & I & J --> K[Validate Mappings]
    K --> L[Compute Confidence]
    L --> M[Cache Mapping]
    M --> N[Return SemanticMapping]
    C --> N
```

## Deployment Architecture

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            ING[Ingress Controller]
        end

        subgraph "PolyglotLink Pods"
            P1[Pod 1]
            P2[Pod 2]
            P3[Pod 3]
        end

        subgraph "Services"
            SVC[ClusterIP Service]
            HPA[Horizontal Pod Autoscaler]
        end

        subgraph "Config"
            CM[ConfigMap]
            SEC[Secrets]
        end
    end

    subgraph "External Services"
        MQTT_EXT[MQTT Broker]
        KAFKA_EXT[Kafka Cluster]
        REDIS_EXT[Redis Cluster]
        PG_EXT[TimescaleDB]
    end

    ING --> SVC
    SVC --> P1 & P2 & P3
    HPA --> SVC
    CM & SEC --> P1 & P2 & P3
    P1 & P2 & P3 --> MQTT_EXT & KAFKA_EXT & REDIS_EXT & PG_EXT
```

### Docker Compose Stack

```mermaid
graph TB
    subgraph "Docker Network"
        subgraph "Application"
            APP[polyglotlink]
        end

        subgraph "Message Brokers"
            MQTT[Mosquitto]
            KAFKA[Kafka]
            ZK[Zookeeper]
        end

        subgraph "Databases"
            REDIS[Redis]
            NEO4J[Neo4j]
            WEAVIATE[Weaviate]
            TIMESCALE[TimescaleDB]
        end

        subgraph "Monitoring"
            PROM[Prometheus]
            GRAF[Grafana]
        end
    end

    APP <--> MQTT
    APP <--> KAFKA
    KAFKA <--> ZK
    APP <--> REDIS
    APP <--> NEO4J
    APP <--> WEAVIATE
    APP <--> TIMESCALE
    PROM --> APP
    GRAF --> PROM
```

## Data Models

### Message Flow Data Structures

```mermaid
classDiagram
    class RawMessage {
        +str message_id
        +str device_id
        +Protocol protocol
        +str topic
        +bytes payload_raw
        +PayloadEncoding payload_encoding
        +datetime timestamp
        +dict metadata
    }

    class ExtractedSchema {
        +str message_id
        +str device_id
        +Protocol protocol
        +str topic
        +list~ExtractedField~ fields
        +str schema_signature
        +dict payload_decoded
        +datetime extracted_at
    }

    class ExtractedField {
        +str key
        +str original_key
        +Any value
        +str value_type
        +str inferred_unit
        +str inferred_semantic
        +bool is_timestamp
        +bool is_identifier
    }

    class SemanticMapping {
        +str message_id
        +str device_id
        +str schema_signature
        +list~FieldMapping~ field_mappings
        +str device_context
        +float confidence
        +bool llm_generated
        +datetime translated_at
    }

    class FieldMapping {
        +str source_field
        +str target_concept
        +str target_field
        +str source_unit
        +str target_unit
        +str conversion_formula
        +float confidence
        +ResolutionMethod resolution_method
    }

    class NormalizedMessage {
        +str message_id
        +str device_id
        +datetime timestamp
        +dict data
        +dict metadata
        +float quality_score
        +datetime normalized_at
    }

    RawMessage --> ExtractedSchema : extract
    ExtractedSchema --> SemanticMapping : translate
    ExtractedSchema --> NormalizedMessage : normalize
    SemanticMapping --> NormalizedMessage : apply
    ExtractedSchema "1" *-- "*" ExtractedField
    SemanticMapping "1" *-- "*" FieldMapping
```

## Caching Strategy

```mermaid
flowchart LR
    subgraph "Cache Layers"
        L1[In-Memory LRU]
        L2[Redis]
        L3[Disk/File]
    end

    subgraph "Cached Data"
        SC[Schema Cache]
        MC[Mapping Cache]
        EC[Embedding Cache]
        OC[Ontology Cache]
    end

    L1 --> L2 --> L3
    SC & MC & EC & OC --> L1
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Error Occurs] --> B{Error Type}

    B -->|Validation Error| C[Log Warning]
    C --> D[Return Partial Result]

    B -->|Parse Error| E[Log Error]
    E --> F[Skip Message]
    F --> G[Increment Error Metric]

    B -->|Connection Error| H[Log Error]
    H --> I{Retries Left?}
    I -->|Yes| J[Exponential Backoff]
    J --> K[Retry Operation]
    I -->|No| L[Circuit Breaker]

    B -->|Critical Error| M[Log Critical]
    M --> N[Send to Sentry]
    N --> O[Alert Operations]

    G & L & O --> P[Update Health Status]
```

## Metrics Collection

```mermaid
flowchart LR
    subgraph "Application"
        M1[Message Metrics]
        M2[Latency Metrics]
        M3[Cache Metrics]
        M4[Error Metrics]
        M5[LLM Metrics]
    end

    subgraph "Prometheus"
        PC[Prometheus Collector]
    end

    subgraph "Visualization"
        G[Grafana]
        A[Alertmanager]
    end

    M1 & M2 & M3 & M4 & M5 --> PC
    PC --> G
    PC --> A
```

## Security Architecture

```mermaid
flowchart TB
    subgraph "External"
        C[Client/Device]
    end

    subgraph "Security Layer"
        TLS[TLS Encryption]
        AUTH[API Key Auth]
        RL[Rate Limiter]
        VAL[Input Validation]
    end

    subgraph "Application"
        APP[PolyglotLink Core]
    end

    subgraph "Secrets Management"
        ENV[Environment Variables]
        SM[Secrets Manager]
        VAULT[HashiCorp Vault]
    end

    C --> TLS
    TLS --> AUTH
    AUTH --> RL
    RL --> VAL
    VAL --> APP

    SM & VAULT & ENV --> APP
```
