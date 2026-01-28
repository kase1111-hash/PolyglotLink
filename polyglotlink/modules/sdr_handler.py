"""
PolyglotLink SDR (Software Defined Radio) Handler Module

This module provides integration with SDR hardware (RTL-SDR, HackRF) for
receiving and decoding radio signals. It supports multiple protocols including:
- ADS-B (aircraft tracking)
- POCSAG (pager messages)
- APRS (amateur packet radio)
- ACARS (aircraft communications)
- RDS (FM broadcast data)
- FLEX (high-speed paging)
"""

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from typing import Any

import structlog

from polyglotlink.models.schemas import (
    PayloadEncoding,
    Protocol,
    RawMessage,
    SDRConfig,
)

logger = structlog.get_logger(__name__)

# Try to import SDR module components
try:
    from polyglotlink.modules.sdr_module.src.sdr_module import (
        DualSDRController,
    )
    from polyglotlink.modules.sdr_module.src.sdr_module.dsp.protocols import (
        ACARSDecoder,
        ADSBDecoder,
        AX25Decoder,
        FLEXDecoder,
        POCSAGDecoder,
        RDSDecoder,
    )
    from polyglotlink.modules.sdr_module.src.sdr_module.dsp.spectrum import (
        SpectrumAnalyzer,
    )

    SDR_AVAILABLE = True
except ImportError as e:
    SDR_AVAILABLE = False
    logger.warning("SDR module not available", error=str(e))

    # Stub classes for type hints when SDR is not available
    class DualSDRController:  # type: ignore[no-redef]
        pass


def generate_uuid() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


class SDRHandler:
    """
    SDR protocol handler for receiving and decoding radio signals.

    Integrates with RTL-SDR and HackRF devices to receive signals and
    decode various protocols like ADS-B, POCSAG, APRS, etc.
    """

    def __init__(self, config: SDRConfig):
        """
        Initialize SDR handler.

        Args:
            config: SDR configuration settings
        """
        self.config = config
        self._running = False
        self._message_queue: asyncio.Queue[RawMessage] = asyncio.Queue()
        self._controller: DualSDRController | None = None
        self._decoders: dict[str, Any] = {}
        self._spectrum_analyzer: Any = None
        self._processing_task: asyncio.Task | None = None

    @property
    def protocol(self) -> Protocol:
        """Get protocol type."""
        return Protocol.SDR

    async def start(self) -> None:
        """Start SDR listener and protocol decoders."""
        if not SDR_AVAILABLE:
            logger.warning("SDR module not installed, handler disabled")
            return

        try:
            logger.info("Initializing SDR handler...")

            # Initialize dual-SDR controller
            self._controller = DualSDRController()

            if not self._controller.initialize():
                logger.warning("No SDR devices found, running in simulation mode")

            # Initialize protocol decoders based on config
            await self._initialize_decoders()

            # Start receiving
            self._running = True
            self._processing_task = asyncio.create_task(self._process_loop())

            logger.info(
                "SDR handler started",
                decoders=list(self._decoders.keys()),
            )

        except Exception as e:
            logger.error("Failed to start SDR handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop SDR listener and cleanup."""
        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        if self._controller:
            self._controller.shutdown()

        self._decoders.clear()
        logger.info("SDR handler stopped")

    async def _initialize_decoders(self) -> None:
        """Initialize protocol decoders based on configuration."""
        decoders = self.config.decoders
        sample_rate = self.config.rtlsdr.sample_rate

        # ADS-B decoder
        if decoders.adsb_enabled:
            try:
                self._decoders["adsb"] = ADSBDecoder(sample_rate=2_000_000)
                logger.info("ADS-B decoder initialized", frequency="1090 MHz")
            except Exception as e:
                logger.warning("Failed to initialize ADS-B decoder", error=str(e))

        # POCSAG decoder
        if decoders.pocsag_enabled:
            try:
                self._decoders["pocsag"] = POCSAGDecoder(sample_rate=sample_rate)
                logger.info(
                    "POCSAG decoder initialized",
                    frequency=f"{decoders.pocsag_frequency / 1e6:.3f} MHz",
                )
            except Exception as e:
                logger.warning("Failed to initialize POCSAG decoder", error=str(e))

        # APRS/AX.25 decoder
        if decoders.aprs_enabled:
            try:
                self._decoders["aprs"] = AX25Decoder(sample_rate=sample_rate)
                logger.info(
                    "APRS decoder initialized",
                    frequency=f"{decoders.aprs_frequency / 1e6:.3f} MHz",
                )
            except Exception as e:
                logger.warning("Failed to initialize APRS decoder", error=str(e))

        # ACARS decoder
        if decoders.acars_enabled:
            try:
                self._decoders["acars"] = ACARSDecoder(sample_rate=sample_rate)
                logger.info("ACARS decoder initialized")
            except Exception as e:
                logger.warning("Failed to initialize ACARS decoder", error=str(e))

        # RDS decoder
        if decoders.rds_enabled:
            try:
                self._decoders["rds"] = RDSDecoder(sample_rate=sample_rate)
                logger.info("RDS decoder initialized")
            except Exception as e:
                logger.warning("Failed to initialize RDS decoder", error=str(e))

        # FLEX decoder
        if decoders.flex_enabled:
            try:
                self._decoders["flex"] = FLEXDecoder(sample_rate=sample_rate)
                logger.info("FLEX decoder initialized")
            except Exception as e:
                logger.warning("Failed to initialize FLEX decoder", error=str(e))

        # Initialize spectrum analyzer if enabled
        if self.config.spectrum_enabled:
            try:
                self._spectrum_analyzer = SpectrumAnalyzer(
                    sample_rate=sample_rate,
                    fft_size=1024,
                )
                logger.info("Spectrum analyzer initialized")
            except Exception as e:
                logger.warning("Failed to initialize spectrum analyzer", error=str(e))

    async def _process_loop(self) -> None:
        """Main processing loop for receiving and decoding signals."""
        while self._running:
            try:
                # Read samples from SDR (if available)
                if self._controller and self._controller.rtlsdr:
                    samples = self._controller.read_rtlsdr_samples(
                        n_samples=self.config.buffer_size,
                        timeout=0.5,
                    )

                    if samples is not None and len(samples) > 0:
                        # Process through each enabled decoder
                        await self._process_samples(samples)

                elif self._controller and self._controller.hackrf:
                    samples = self._controller.read_hackrf_samples(
                        n_samples=self.config.buffer_size,
                        timeout=0.5,
                    )

                    if samples is not None and len(samples) > 0:
                        await self._process_samples(samples)
                else:
                    # No hardware available, sleep briefly
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in SDR processing loop", error=str(e))
                await asyncio.sleep(0.5)

    async def _process_samples(self, samples: Any) -> None:
        """Process IQ samples through decoders."""
        # Spectrum analysis (for signal visualization)
        if self._spectrum_analyzer:
            try:
                self._spectrum_analyzer.compute_spectrum(samples)
            except Exception as e:
                logger.debug("Spectrum analysis error", error=str(e))

        # Process through each decoder
        for decoder_name, decoder in self._decoders.items():
            try:
                messages = decoder.decode(samples)

                for msg in messages:
                    raw_message = await self._convert_to_raw_message(decoder_name, msg)
                    if raw_message:
                        await self._message_queue.put(raw_message)

            except Exception as e:
                logger.debug(
                    "Decoder error",
                    decoder=decoder_name,
                    error=str(e),
                )

    async def _convert_to_raw_message(
        self, decoder_name: str, decoded_msg: Any
    ) -> RawMessage | None:
        """Convert decoded SDR message to RawMessage format."""
        try:
            # Map decoder to payload encoding
            encoding_map = {
                "adsb": PayloadEncoding.SDR_ADSB,
                "pocsag": PayloadEncoding.SDR_POCSAG,
                "aprs": PayloadEncoding.SDR_APRS,
                "acars": PayloadEncoding.SDR_ACARS,
                "rds": PayloadEncoding.SDR_RDS,
                "flex": PayloadEncoding.SDR_FLEX,
            }

            encoding = encoding_map.get(decoder_name, PayloadEncoding.SDR_IQ)

            # Extract device ID based on message type
            device_id = self._extract_device_id(decoder_name, decoded_msg)

            # Convert message to JSON payload
            payload_dict = self._message_to_dict(decoder_name, decoded_msg)
            payload_bytes = json.dumps(payload_dict).encode("utf-8")

            # Create topic based on decoder and device
            topic = f"sdr/{decoder_name}/{device_id}"

            return RawMessage(
                message_id=generate_uuid(),
                device_id=device_id,
                protocol=Protocol.SDR,
                topic=topic,
                payload_raw=payload_bytes,
                payload_encoding=encoding,
                timestamp=datetime.utcnow(),
                metadata={
                    "sdr_decoder": decoder_name,
                    "signal_timestamp": getattr(decoded_msg, "timestamp", 0),
                    "valid": getattr(decoded_msg, "valid", True),
                },
            )

        except Exception as e:
            logger.error("Failed to convert SDR message", error=str(e))
            return None

    def _extract_device_id(self, decoder_name: str, msg: Any) -> str:
        """Extract device identifier from decoded message."""
        if decoder_name == "adsb":
            return getattr(msg, "icao_address", "unknown")
        elif decoder_name == "pocsag":
            return f"pager_{getattr(msg, 'address', 'unknown')}"
        elif decoder_name == "aprs":
            return getattr(msg, "source", "unknown")
        elif decoder_name == "acars":
            return getattr(msg, "registration", "unknown").strip()
        elif decoder_name == "rds":
            return f"station_{getattr(msg, 'pi_code', 0):04X}"
        elif decoder_name == "flex":
            return f"flex_{getattr(msg, 'capcode', 'unknown')}"
        else:
            return "unknown"

    def _message_to_dict(self, decoder_name: str, msg: Any) -> dict[str, Any]:
        """Convert decoded message to dictionary."""
        result: dict[str, Any] = {
            "decoder": decoder_name,
            "timestamp": getattr(msg, "timestamp", 0),
            "valid": getattr(msg, "valid", True),
        }

        if decoder_name == "adsb":
            result.update({
                "icao": getattr(msg, "icao_address", ""),
                "callsign": getattr(msg, "callsign", ""),
                "latitude": getattr(msg, "latitude", 0.0),
                "longitude": getattr(msg, "longitude", 0.0),
                "altitude_ft": getattr(msg, "altitude", 0),
                "velocity_kts": getattr(msg, "velocity", 0.0),
                "heading": getattr(msg, "heading", 0.0),
                "vertical_rate_fpm": getattr(msg, "vertical_rate", 0),
                "on_ground": getattr(msg, "on_ground", False),
                "category": getattr(msg, "category", ""),
            })

        elif decoder_name == "pocsag":
            result.update({
                "address": getattr(msg, "address", 0),
                "function": getattr(msg, "function", 0),
                "message_type": getattr(msg, "message_type", ""),
                "content": getattr(msg, "content", ""),
                "baud_rate": getattr(msg, "baud_rate", 1200),
            })

        elif decoder_name == "aprs":
            result.update({
                "source": getattr(msg, "source", ""),
                "destination": getattr(msg, "destination", ""),
                "path": getattr(msg, "digipeaters", []) or getattr(msg, "path", []),
                "data_type": getattr(msg, "data_type", ""),
                "latitude": getattr(msg, "latitude", 0.0),
                "longitude": getattr(msg, "longitude", 0.0),
                "comment": getattr(msg, "comment", "") or getattr(msg, "info", ""),
            })

        elif decoder_name == "acars":
            result.update({
                "mode": getattr(msg, "mode", ""),
                "registration": getattr(msg, "registration", ""),
                "flight_id": getattr(msg, "flight_id", ""),
                "label": getattr(msg, "label", ""),
                "text": getattr(msg, "text", ""),
            })

        elif decoder_name == "rds":
            result.update({
                "pi_code": getattr(msg, "pi_code", 0),
                "pty": getattr(msg, "pty", 0),
                "ps_name": getattr(msg, "ps_name", ""),
                "radio_text": getattr(msg, "radio_text", ""),
                "traffic_program": getattr(msg, "tp", False),
                "traffic_announcement": getattr(msg, "ta", False),
            })

        elif decoder_name == "flex":
            result.update({
                "capcode": getattr(msg, "capcode", 0),
                "cycle": getattr(msg, "cycle", 0),
                "frame": getattr(msg, "frame", 0),
                "phase": getattr(msg, "phase", ""),
                "message_type": getattr(msg, "message_type", ""),
                "content": getattr(msg, "content", ""),
            })

        return result

    async def messages(self):
        """Yield messages from the handler."""
        while self._running or not self._message_queue.empty():
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                yield message
            except TimeoutError:
                continue

    async def emit_message(self, message: RawMessage) -> None:
        """Add a message to the queue."""
        await self._message_queue.put(message)

    def get_status(self) -> dict[str, Any]:
        """Get SDR handler status."""
        status = {
            "running": self._running,
            "sdr_available": SDR_AVAILABLE,
            "decoders": list(self._decoders.keys()),
        }

        if self._controller:
            status["controller_status"] = self._controller.get_status()

        return status
