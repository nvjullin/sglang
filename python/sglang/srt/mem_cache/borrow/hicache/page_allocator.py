from __future__ import annotations

from typing import Iterable, Optional


class PageAllocator:
    """CPU free-list over page ids ``[first_page, first_page + num_pages)``.

    Deterministic: the same call sequence yields the same pages on every rank.
    """

    def __init__(self, *, num_pages: int, first_page: int = 0):
        self.num_pages = num_pages
        self.first_page = first_page
        # Popped from the end, so the lowest page ids go out first.
        self._free = list(range(first_page + num_pages - 1, first_page - 1, -1))
        self._is_free = bytearray(b"\x01") * num_pages

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> Optional[list[int]]:
        if n > len(self._free):
            return None
        if n == 0:
            return []
        pages = self._free[-n:]
        del self._free[-n:]
        pages.reverse()
        for p in pages:
            self._is_free[p - self.first_page] = 0
        return pages

    def free(self, pages: Iterable[int]) -> None:
        for p in pages:
            i = p - self.first_page
            if not 0 <= i < self.num_pages or self._is_free[i]:
                raise RuntimeError(f"double or foreign free of page {p}")
            self._is_free[i] = 1
            self._free.append(p)

    def clear(self) -> None:
        self.__init__(num_pages=self.num_pages, first_page=self.first_page)
