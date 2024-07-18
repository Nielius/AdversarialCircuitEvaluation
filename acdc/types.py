from typing import TypeAlias

EdgeAsTuple: TypeAlias = tuple[
    str, tuple[int | None, ...], str, tuple[int | None, ...]
]  # child first, parent second; the child is computationally dependent on the parent
