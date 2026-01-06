from __future__ import annotations

from dataclasses import dataclass

from ..memory import list_adapters, set_active_adapter, rollback_adapter


@dataclass
class AdapterInfo:
    name: str
    path: str
    active: bool


def get_adapters(conn) -> list[AdapterInfo]:
    rows = list_adapters(conn)
    return [AdapterInfo(name=row[0], path=row[1], active=bool(row[2])) for row in rows]


def activate_adapter(conn, name: str) -> None:
    set_active_adapter(conn, name)


def deactivate_adapter(conn, name: str) -> None:
    rollback_adapter(conn, name)
