"""Декораторы для ДЗ 2.

Задание 2.1:
- validate_types - обязательный для всех
- Два дополнительных декоратора определяются вариантом студента

Все декораторы должны:
- Быть параметризованными (фабрика)
- Использовать functools.wraps
- Быть композируемыми (stackable)
- Иметь type hints
"""

from __future__ import annotations

import inspect
import time
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# validate_types - обязательный для всех вариантов
# ============================================================


def validate_types(func: F) -> F:
    """Декоратор, проверяющий типы аргументов и возвращаемого значения по аннотациям.

    Если аргумент или возвращаемое значение не соответствует аннотации,
    выбрасывает TypeError с понятным сообщением.
    Параметры без аннотаций не проверяются.

    Пример::

        @validate_types
        def add(a: int, b: int) -> int:
            return a + b

        add(1, 2)       # OK -> 3
        add(1, "two")   # TypeError
    """
    signature = inspect.signature(func)
    hints = _resolve_hints(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        _check_arguments(signature, hints, bound)
        result = func(*args, **kwargs)
        if "return" in hints and not _matches(result, hints["return"]):
            raise TypeError(
                f"return value expected {_type_name(hints['return'])}, got {type(result).__name__}"
            )
        return result

    return cast(F, wrapper)


def _check_arguments(
    signature: inspect.Signature,
    hints: dict[str, Any],
    bound: inspect.BoundArguments,
) -> None:
    for name, value in bound.arguments.items():
        if name not in hints:
            continue
        annotation = hints[name]
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_POSITIONAL:
            values: tuple[Any, ...] = tuple(value)
        elif kind is inspect.Parameter.VAR_KEYWORD:
            values = tuple(value.values())
        else:
            values = (value,)
        for item in values:
            if not _matches(item, annotation):
                raise TypeError(
                    f"argument {name!r} expected {_type_name(annotation)}, "
                    f"got {type(item).__name__}"
                )


def _matches(value: object, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(_matches(value, argument) for argument in get_args(annotation))
    # у list[int] и прочих дженериков проверяем только сам контейнер
    expected = annotation if origin is None else origin
    if not isinstance(expected, type):
        return True
    return isinstance(value, expected)


def _resolve_hints(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(func)
    except (NameError, TypeError):
        # аннотацию не разрезолвили - просто не проверяем, падать тут незачем
        return dict(getattr(func, "__annotations__", {}))


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


# ============================================================
# Вариант 0, 4: curry
# ============================================================


def curry(func: F) -> F:
    """Декоратор каррирования.

    Позволяет вызывать функцию с частичным применением аргументов::

        @curry
        def add(a: int, b: int, c: int) -> int:
            return a + b + c

        add(1)(2)(3)     # 6
        add(1, 2)(3)     # 6
        add(1)(2, 3)     # 6
    """
    # не входит в вариант 3
    raise NotImplementedError("curry не реализован")


# ============================================================
# Вариант 0, 3: memoize
# ============================================================


def memoize(*, ttl: float | None = None) -> Callable[[F], F]:
    """Декоратор кеширования результатов.

    Args:
        ttl: Время жизни кеша в секундах. None - бессрочно.

    Пример::

        @memoize(ttl=60)
        def expensive(n: int) -> int:
            return n ** 2
    """
    if ttl is not None and ttl <= 0:
        raise ValueError("ttl must be positive")

    def decorator(func: F) -> F:
        cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = args + tuple(sorted(kwargs.items()))
            try:
                hash(key)
            except TypeError:
                # аргументы не хешируются - считаем мимо кеша
                return func(*args, **kwargs)

            entry = cache.get(key)
            if entry is not None and entry[0] > time.monotonic():
                return entry[1]

            result = func(*args, **kwargs)
            expires_at = float("inf") if ttl is None else time.monotonic() + ttl
            cache[key] = (expires_at, result)
            return result

        return cast(F, wrapper)

    return decorator


# ============================================================
# Вариант 1, 3: retry
# ============================================================


def retry(
    *,
    max_retries: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    backoff: float = 1.0,
) -> Callable[[F], F]:
    """Декоратор повторных попыток при исключении.

    Args:
        max_retries: Максимальное количество повторных попыток.
        exceptions: Кортеж классов исключений, при которых повторять.
        backoff: Базовая задержка (экспоненциальная: backoff * 2^attempt).

    Пример::

        @retry(max_retries=3, exceptions=(ConnectionError,), backoff=0.1)
        def fetch(url: str) -> str:
            ...
    """
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")
    if backoff < 0:
        raise ValueError("backoff must not be negative")

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    time.sleep(backoff * 2**attempt)
            # последнюю попытку не перехватываем - её исключение уходит наружу
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# ============================================================
# Вариант 2, 5: deprecated
# ============================================================


def deprecated(*, message: str = "", removal_version: str | None = None) -> Callable[[F], F]:
    """Декоратор, помечающий функцию как устаревшую.

    При вызове выдаёт DeprecationWarning с указанным сообщением.

    Args:
        message: Текст предупреждения.
        removal_version: Версия, в которой функция будет удалена (добавляется в предупреждение).

    Пример::

        @deprecated(message="Используйте new_func вместо old_func", removal_version="2.0")
        def old_func() -> None:
            ...
    """

    # не входит в вариант 3
    def decorator(func: F) -> F:
        raise NotImplementedError("deprecated не реализован")

    return decorator


# ============================================================
# Вариант 1, 4: trace
# ============================================================


def trace(*, logger_name: str = __name__) -> Callable[[F], F]:
    """Декоратор трассировки вызовов через модуль logging.

    Логирует имя функции, аргументы, результат и время выполнения.

    Args:
        logger_name: Имя логгера.

    Пример::

        @trace(logger_name="myapp")
        def compute(x: int) -> int:
            return x * 2
    """

    # не входит в вариант 3
    def decorator(func: F) -> F:
        raise NotImplementedError("trace не реализован")

    return decorator


# ============================================================
# Вариант 2, 5: throttle
# ============================================================


def throttle(*, rate: float) -> Callable[[F], F]:
    """Декоратор ограничения частоты вызовов.

    Если функция вызывается чаще чем раз в rate секунд,
    блокирует (sleep) до истечения интервала.

    Args:
        rate: Минимальный интервал между вызовами в секундах.

    Пример::

        @throttle(rate=1.0)
        def api_call() -> dict:
            ...
    """

    # не входит в вариант 3
    def decorator(func: F) -> F:
        raise NotImplementedError("throttle не реализован")

    return decorator
