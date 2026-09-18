"""Функциональный пайплайн для ДЗ 2.

Задание 2.2:
- pipe(*fns) - применяет функции слева направо
- compose(*fns) - применяет функции справа налево
- filter_by(**kwargs) - фильтрация по атрибутам/ключам
- sort_by(key) - сортировка по ключу
- take(n) - первые N элементов
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def pipe(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """Применяет функции слева направо.

    Пример::

        pipe(str.upper, str.strip)("  hello  ")  # "HELLO"
        pipe(add_one, double)(5)  # double(add_one(5)) = 12
    """

    def piped(value: Any) -> Any:
        for fn in fns:
            value = fn(value)
        return value

    return piped


def compose(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """Применяет функции справа налево.

    Пример::

        compose(double, add_one)(5)  # double(add_one(5)) = 12
    """
    return pipe(*reversed(fns))


def filter_by(**kwargs: Any) -> Callable[[list[Any]], list[Any]]:
    """Возвращает функцию, фильтрующую список словарей/объектов по указанным полям.

    Пример::

        users = [{"name": "Alice", "active": True}, {"name": "Bob", "active": False}]
        filter_by(active=True)(users)  # [{"name": "Alice", "active": True}]
    """

    def matches(item: Any) -> bool:
        return all(_field(item, name) == expected for name, expected in kwargs.items())

    def apply(items: list[Any]) -> list[Any]:
        return [item for item in items if matches(item)]

    return apply


def sort_by(key: str, *, reverse: bool = False) -> Callable[[list[Any]], list[Any]]:
    """Возвращает функцию, сортирующую список словарей/объектов по указанному ключу.

    Пример::

        sort_by("name")(users)  # отсортировано по name
    """

    def apply(items: list[Any]) -> list[Any]:
        return sorted(items, key=lambda item: _field(item, key), reverse=reverse)

    return apply


def take(n: int) -> Callable[[list[Any]], list[Any]]:
    """Возвращает функцию, берущую первые n элементов.

    Пример::

        take(2)([1, 2, 3, 4])  # [1, 2]
    """
    if n < 0:
        raise ValueError("n must not be negative")

    def apply(items: list[Any]) -> list[Any]:
        return list(items[:n])

    return apply


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)
