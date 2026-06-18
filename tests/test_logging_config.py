"""Central logging policy tests."""

from __future__ import annotations

import logging


def test_logging_config_filters_benign_asyncio_socket_send_warning() -> None:
    """Closed browser sockets should not flood production stderr."""

    from kazusa_ai_chatbot import logging_config

    logging_config._configured_profile = None
    logging_config.configure_service_logging()
    asyncio_logger = logging.getLogger("asyncio")
    benign_record = asyncio_logger.makeRecord(
        "asyncio",
        logging.WARNING,
        __file__,
        1,
        "socket.send() raised exception.",
        (),
        None,
    )
    real_record = asyncio_logger.makeRecord(
        "asyncio",
        logging.WARNING,
        __file__,
        1,
        "socket transport failed: real failure",
        (),
        None,
    )

    assert not asyncio_logger.filter(benign_record)
    assert asyncio_logger.filter(real_record)
