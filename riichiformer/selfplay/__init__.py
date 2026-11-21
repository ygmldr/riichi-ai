"""PR-41/PR-42 local self-play: heuristic bots, rollout provenance and seed pools.

This package never contacts an online platform, never downloads a log or a weight and
never claims Tenhou parity or a rank.  The real-rule path is the pinned local
``libriichi`` arena; the procedural fallback is explicitly labelled synthetic.
"""

from __future__ import annotations

from .bots import (
    BOT_FACTORIES,
    BOT_VERSION,
    Bot,
    BotAction,
    BotError,
    DefenseBot,
    IllegalBotAction,
    LegalRandomBot,
    SimpleShantenUkeireBot,
    bot_specs,
    default_lineup,
    make_bot,
)
from .view import SeatView, ViewError

__all__ = [
    "BOT_FACTORIES",
    "BOT_VERSION",
    "Bot",
    "BotAction",
    "BotError",
    "DefenseBot",
    "IllegalBotAction",
    "LegalRandomBot",
    "SeatView",
    "SimpleShantenUkeireBot",
    "ViewError",
    "bot_specs",
    "default_lineup",
    "make_bot",
]
