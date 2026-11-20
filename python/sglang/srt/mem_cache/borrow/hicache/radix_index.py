"""Page-granular radix index over cached KV, with a device and a host tier.

Invariants:
- A node's ``tokens`` length is a positive multiple of ``page_size``.
- Device residency is prefix-closed: a device-resident node's parent is
  device-resident (the root always is).
- ``lock_ref`` counts leases whose shared prefix ends at or below the node;
  only device-resident nodes are ever locked.
- Recency uses a logical clock, so every replica evicts the same nodes.
- A node's recurrent ``state`` (device slot) / ``host_state`` (host slot) is
  the state after the node's last token; a device state needs device pages and
  a host state needs host pages. States carry no locks: every reader of a
  state slot is ordered before its reuse on the service's state stream.
- With a sliding window, each device page may have a window page (None: a
  tombstone, its window KV is gone); likewise on host. Leases pin the window
  pages they read by page id, so pins survive node splits; eviction only
  tombstones unpinned window pages.
"""

from __future__ import annotations

import heapq
from typing import Any, Hashable, Optional, Sequence

import msgspec


class Node:
    __slots__ = (
        "id",
        "parent",
        "children",
        "tokens",
        "device_pages",
        "host_pages",
        "lock_ref",
        "last_access",
        "d2h_op",
        "h2d_op",
        "state",
        "host_state",
        "window_pages",
        "host_window_pages",
    )

    def __init__(self, *, id: int, parent: Optional[Node], tokens: tuple):
        self.id = id
        self.parent = parent
        self.children: dict[Hashable, Node] = {}
        self.tokens = tokens
        self.device_pages: Optional[list[int]] = None
        self.host_pages: Optional[list[int]] = None
        self.lock_ref = 0
        self.last_access = 0
        # Transfer ops the node's pages take part in; see data_plane.TransferOp.
        self.d2h_op: Any = None
        self.h2d_op: Any = None
        self.state: Optional[int] = None
        self.host_state: Optional[int] = None
        # Per page of device_pages / host_pages: window page or None.
        self.window_pages: Optional[list] = None
        self.host_window_pages: Optional[list] = None

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id}, len={len(self.tokens)}, dev={self.device_pages is not None}, "
            f"host={self.host_pages is not None}, lock={self.lock_ref}, "
            f"state={self.state}, host_state={self.host_state})"
        )


class MatchPath(msgspec.Struct):
    # Fully matched nodes, root excluded, in root-to-leaf order.
    nodes: list
    device_len: int
    host_len: int


class Evicted(msgspec.Struct):
    pages: list
    # Recurrent-state slots the evicted nodes held (device or host tier).
    states: list
    # Window pages the evicted nodes held (device or host tier).
    windows: list = []


class InsertOutcome(msgspec.Struct):
    last_node: Any
    # Per page of the inserted span: True if the index now owns the caller's page.
    adopted: list
    # Nodes that became device-resident with the caller's pages.
    new_resident: list
    # Cached device pages the caller's pages replaced (window recovery); the
    # caller frees them.
    replaced: list = []


class PrefixIndex:
    def __init__(self, *, page_size: int):
        self.page_size = page_size
        self._next_id = 1
        self.root = Node(id=0, parent=None, tokens=())
        self.root.device_pages = []
        self.root.host_pages = []
        self.clock = 0
        self.device_pages_cached = 0
        self.evictable_device_pages = 0
        self.host_pages_cached = 0
        self._device_leaves: set[Node] = set()
        self._host_leaves: set[Node] = set()
        self._state_nodes: set[Node] = set()
        self._host_state_nodes: set[Node] = set()
        self._window_nodes: set[Node] = set()
        self._host_window_nodes: set[Node] = set()
        self.window_pages_cached = 0
        # Window page id -> number of leases pinning it.
        self._window_pins: dict[int, int] = {}

    # ===== Lookup =====

    def _child_key(self, node: Node, namespace: tuple, tokens: Sequence[int], pos: int):
        page = tuple(tokens[pos : pos + self.page_size])
        return (namespace, page) if node is self.root else page

    def _common_len(self, node_tokens: tuple, tokens: Sequence[int], pos: int) -> int:
        n = min(len(node_tokens), len(tokens) - pos)
        if tuple(tokens[pos : pos + n]) == node_tokens[:n]:
            common = n
        else:
            common = 0
            while common < n and node_tokens[common] == tokens[pos + common]:
                common += 1
        return common // self.page_size * self.page_size

    def _usable_len(self, tokens: Sequence[int]) -> int:
        return len(tokens) // self.page_size * self.page_size

    def peek(
        self, namespace: tuple, tokens: Sequence[int], *, touch: bool = False
    ) -> tuple[int, int]:
        """(device_len, cached_len) of the longest cached prefix. No structural
        change; ``touch`` refreshes the recency of fully matched nodes."""
        tokens = tokens[: self._usable_len(tokens)]
        if touch:
            self.clock += 1
        node, pos, device_len = self.root, 0, 0
        while pos < len(tokens):
            child = node.children.get(self._child_key(node, namespace, tokens, pos))
            if child is None:
                break
            common = self._common_len(child.tokens, tokens, pos)
            if common == 0:
                break
            if child.device_pages is not None and device_len == pos:
                device_len = pos + common
            pos += common
            if common < len(child.tokens):
                break
            if touch:
                child.last_access = self.clock
            node = child
        return device_len, pos

    def match(self, namespace: tuple, tokens: Sequence[int]) -> MatchPath:
        """Walk the longest cached prefix, splitting a partially matched node."""
        tokens = tokens[: self._usable_len(tokens)]
        self.clock += 1
        node, pos = self.root, 0
        nodes: list[Node] = []
        while pos < len(tokens):
            child = node.children.get(self._child_key(node, namespace, tokens, pos))
            if child is None:
                break
            common = self._common_len(child.tokens, tokens, pos)
            if common == 0:
                break
            if common < len(child.tokens):
                child = self._split(child, common)
            child.last_access = self.clock
            nodes.append(child)
            pos += common
            node = child
        device_len = 0
        for n in nodes:
            if n.device_pages is None:
                break
            device_len += len(n.tokens)
        return MatchPath(nodes=nodes, device_len=device_len, host_len=pos - device_len)

    # ===== Structure =====

    def _new_node(self, parent: Node, tokens: tuple) -> Node:
        node = Node(id=self._next_id, parent=parent, tokens=tokens)
        self._next_id += 1
        return node

    def _split(self, node: Node, at: int) -> Node:
        """Split ``node`` so that a new parent covers its first ``at`` tokens."""
        assert 0 < at < len(node.tokens) and at % self.page_size == 0
        parent = node.parent
        key = self._key_in_parent(node)
        head = self._new_node(parent, node.tokens[:at])
        cut = at // self.page_size
        if node.device_pages is not None:
            head.device_pages = node.device_pages[:cut]
            node.device_pages = node.device_pages[cut:]
        if node.host_pages is not None:
            head.host_pages = node.host_pages[:cut]
            node.host_pages = node.host_pages[cut:]
        if node.window_pages is not None:
            head.window_pages = node.window_pages[:cut]
            node.window_pages = node.window_pages[cut:]
        if node.host_window_pages is not None:
            head.host_window_pages = node.host_window_pages[:cut]
            node.host_window_pages = node.host_window_pages[cut:]
        self._track_windows(head)
        self._track_windows(node)
        head.lock_ref = node.lock_ref
        head.last_access = node.last_access
        head.d2h_op, head.h2d_op = node.d2h_op, node.h2d_op
        # A state is the state after the node's last token, so it stays with
        # the tail.
        node.tokens = node.tokens[at:]
        node.parent = head
        parent.children[key] = head
        head.children[self._page_key(node.tokens)] = node
        self._update_leaf_sets(head)
        self._update_leaf_sets(node)
        return head

    def _page_key(self, tokens: tuple) -> tuple:
        return tuple(tokens[: self.page_size])

    def _key_in_parent(self, node: Node):
        for key, child in node.parent.children.items():
            if child is node:
                return key
        raise RuntimeError(f"{node} is not a child of its parent")

    def _update_leaf_sets(self, node: Optional[Node]) -> None:
        if node is None or node is self.root:
            return
        is_device_leaf = node.device_pages is not None and not any(
            c.device_pages is not None for c in node.children.values()
        )
        if is_device_leaf:
            self._device_leaves.add(node)
        else:
            self._device_leaves.discard(node)
        is_host_leaf = (
            node.device_pages is None
            and node.host_pages is not None
            and not node.children
        )
        if is_host_leaf:
            self._host_leaves.add(node)
        else:
            self._host_leaves.discard(node)

    def _set_device_pages(self, node: Node, pages: Optional[list[int]]) -> None:
        old = len(node.device_pages) if node.device_pages is not None else 0
        new = len(pages) if pages is not None else 0
        node.device_pages = pages
        self.device_pages_cached += new - old
        if node.lock_ref == 0:
            self.evictable_device_pages += new - old
        self._update_leaf_sets(node)
        self._update_leaf_sets(node.parent)

    def _set_host_pages(self, node: Node, pages: Optional[list[int]]) -> None:
        old = len(node.host_pages) if node.host_pages is not None else 0
        new = len(pages) if pages is not None else 0
        node.host_pages = pages
        self.host_pages_cached += new - old
        self._update_leaf_sets(node)

    def _remove_leaf(self, node: Node) -> None:
        assert not node.children and node.lock_ref == 0
        parent = node.parent
        del parent.children[self._key_in_parent(node)]
        self._device_leaves.discard(node)
        self._host_leaves.discard(node)
        self._window_nodes.discard(node)
        self._host_window_nodes.discard(node)
        self._update_leaf_sets(parent)

    # ===== Locks =====

    def lock(self, node: Node) -> None:
        while node is not self.root:
            if node.lock_ref == 0:
                assert node.device_pages is not None, f"locking non-resident {node}"
                self.evictable_device_pages -= len(node.device_pages)
            node.lock_ref += 1
            node = node.parent

    def unlock(self, node: Node) -> None:
        while node is not self.root:
            assert node.lock_ref > 0, f"unbalanced unlock of {node}"
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self.evictable_device_pages += len(node.device_pages)
            node = node.parent

    # ===== Mutation by leases =====

    def commit_load(
        self,
        node: Node,
        device_pages: list[int],
        op: Any,
        window_pages: Optional[list] = None,
    ) -> None:
        """Make a host-only node device-resident; ``op`` fills its pages."""
        assert node.device_pages is None and node.host_pages is not None
        assert node.parent.device_pages is not None
        node.h2d_op = op
        self._set_device_pages(node, device_pages)
        self._set_window_pages(node, window_pages)

    def insert(
        self,
        namespace: tuple,
        tokens: Sequence[int],
        pages: Sequence[int],
        window_pages: Optional[Sequence] = None,
    ) -> InsertOutcome:
        """Cache ``tokens`` backed by ``pages`` (one per page of tokens) and,
        with a sliding window, ``window_pages`` (None entries: tombstones).
        A node that takes the caller's pages takes their window pages too, and
        an unlocked cached page whose window is a tombstone is swapped for the
        caller's page and its live window page (window recovery)."""
        ps = self.page_size
        assert len(tokens) % ps == 0 and len(pages) == len(tokens) // ps
        self.clock += 1
        adopted = [False] * len(pages)
        new_resident: list[Node] = []
        replaced: list[int] = []
        node, pos = self.root, 0
        while pos < len(tokens):
            child = node.children.get(self._child_key(node, namespace, tokens, pos))
            if child is None:
                child = self._new_node(node, tuple(tokens[pos:]))
                node.children[self._child_key(node, namespace, tokens, pos)] = child
                self._take_pages(
                    child, pages, window_pages, pos // ps, adopted, new_resident
                )
                child.last_access = self.clock
                node = child
                break
            common = self._common_len(child.tokens, tokens, pos)
            assert common > 0
            if common < len(child.tokens):
                child = self._split(child, common)
            if child.device_pages is None:
                self._take_pages(
                    child, pages, window_pages, pos // ps, adopted, new_resident
                )
            elif window_pages is not None:
                self._recover_windows(child, pages, window_pages, pos // ps, adopted, replaced)
            child.last_access = self.clock
            pos += common
            node = child
        return InsertOutcome(
            last_node=node, adopted=adopted, new_resident=new_resident, replaced=replaced
        )

    def _recover_windows(
        self,
        node: Node,
        pages: Sequence[int],
        window_pages: Sequence,
        first: int,
        adopted: list[bool],
        replaced: list[int],
    ) -> None:
        """Swap tombstoned pages of an unborrowed cached node for the caller's
        duplicate pages that still have window KV."""
        if node.lock_ref or (node.h2d_op is not None and not node.h2d_op.launched):
            return
        windows = node.window_pages or [None] * len(node.device_pages)
        swap = [
            j
            for j, (old, w) in enumerate(zip(node.device_pages, windows))
            if w is None
            and window_pages[first + j] is not None
            and pages[first + j] != old
        ]
        if not swap:
            return
        if node.d2h_op is not None:
            # The old pages may still be read by their backup.
            node.d2h_op.wait()
        device, windows = list(node.device_pages), list(windows)
        for j in swap:
            replaced.append(device[j])
            device[j], windows[j] = pages[first + j], window_pages[first + j]
            adopted[first + j] = True
        node.device_pages = device
        self._set_window_pages(node, windows)

    def _take_pages(
        self,
        node: Node,
        pages: Sequence[int],
        window_pages: Optional[Sequence],
        first: int,
        adopted: list[bool],
        new_resident: list[Node],
    ) -> None:
        n = len(node.tokens) // self.page_size
        self._set_device_pages(node, list(pages[first : first + n]))
        if window_pages is not None:
            self._set_window_pages(node, list(window_pages[first : first + n]))
        for i in range(first, first + n):
            adopted[i] = True
        new_resident.append(node)

    def set_backup(
        self,
        node: Node,
        host_pages: list[int],
        op: Any,
        host_window_pages: Optional[list] = None,
    ) -> None:
        assert node.device_pages is not None and node.host_pages is None
        node.d2h_op = op
        self._set_host_pages(node, host_pages)
        node.host_window_pages = host_window_pages
        self._track_windows(node)

    def unbacked_on_path(self, node: Node) -> list[Node]:
        """Device-resident, unbacked nodes from the root down to ``node``."""
        out = []
        while node is not self.root:
            if node.host_pages is None:
                out.append(node)
            node = node.parent
        out.reverse()
        return out

    # ===== Recurrent state =====

    def node_at(self, namespace: tuple, tokens: Sequence[int]) -> Node:
        """The cached node ending exactly after ``tokens`` (page-aligned),
        splitting one if needed."""
        assert tokens and len(tokens) % self.page_size == 0
        node, pos = self.root, 0
        while pos < len(tokens):
            child = node.children[self._child_key(node, namespace, tokens, pos)]
            common = self._common_len(child.tokens, tokens, pos)
            assert common > 0, f"{len(tokens)} tokens are not cached"
            if common < len(child.tokens):
                child = self._split(child, common)
            pos += common
            node = child
        return node

    def set_state(self, node: Node, slot: int) -> None:
        assert node.state is None and node.device_pages is not None
        node.state = slot
        self._state_nodes.add(node)

    def set_host_state(self, node: Node, slot: int) -> None:
        assert node.host_state is None and node.host_pages is not None
        node.host_state = slot
        self._host_state_nodes.add(node)

    def _drop_state(self, node: Node, out: list[int]) -> None:
        if node.state is not None:
            out.append(node.state)
            node.state = None
            self._state_nodes.discard(node)

    def _drop_host_state(self, node: Node, out: list[int]) -> None:
        if node.host_state is not None:
            out.append(node.host_state)
            node.host_state = None
            self._host_state_nodes.discard(node)

    @property
    def num_states(self) -> int:
        return len(self._state_nodes)

    def evict_states(self, num: int, *, keep: Optional[Node] = None) -> list[int]:
        """Free up to ``num`` device state slots, LRU first; nodes keep their KV."""
        order = sorted(
            (n.last_access, n.id, n) for n in self._state_nodes if n is not keep
        )
        out: list[int] = []
        for _, _, node in order[:num]:
            self._drop_state(node, out)
        return out

    def evict_host_states(self, num: int) -> list[int]:
        """Free up to ``num`` host state slots, LRU first."""
        order = sorted((n.last_access, n.id, n) for n in self._host_state_nodes)
        out: list[int] = []
        for _, _, node in order[:num]:
            self._drop_host_state(node, out)
        return out

    # ===== Sliding window =====

    @staticmethod
    def live_windows(pages: Optional[list]) -> list[int]:
        return [] if pages is None else [w for w in pages if w is not None]

    def _track_windows(self, node: Node) -> None:
        for pages, nodes in (
            (node.window_pages, self._window_nodes),
            (node.host_window_pages, self._host_window_nodes),
        ):
            if self.live_windows(pages):
                nodes.add(node)
            else:
                nodes.discard(node)

    def _set_window_pages(self, node: Node, pages: Optional[list]) -> None:
        delta = len(self.live_windows(pages)) - len(self.live_windows(node.window_pages))
        self.window_pages_cached += delta
        node.window_pages = pages
        self._track_windows(node)

    def _drop_windows(self, node: Node, out: list[int]) -> None:
        live = self.live_windows(node.window_pages)
        assert not any(w in self._window_pins for w in live), f"evicting pinned windows of {node}"
        out.extend(live)
        self._set_window_pages(node, None)

    def _drop_host_windows(self, node: Node, out: list[int]) -> None:
        out.extend(self.live_windows(node.host_window_pages))
        node.host_window_pages = None
        self._track_windows(node)

    def pin_windows(self, pages: Sequence[int]) -> None:
        for w in pages:
            self._window_pins[w] = self._window_pins.get(w, 0) + 1

    def unpin_windows(self, pages: Sequence[int]) -> None:
        for w in pages:
            left = self._window_pins[w] - 1
            if left:
                self._window_pins[w] = left
            else:
                del self._window_pins[w]

    @property
    def evictable_window_pages(self) -> int:
        # Only cached window pages are ever pinned.
        return self.window_pages_cached - len(self._window_pins)

    def evict_windows(self, num: int, *, trailing_pages: int = 0) -> list[int]:
        """Tombstone up to ``num`` unpinned device window pages; nodes keep
        their KV pages. A node's last ``trailing_pages`` window pages are what
        a hit ending at the node reads, so every node's other (interior) pages
        go first, LRU nodes first within each pass."""
        order = [n for _, _, n in sorted((n.last_access, n.id, n) for n in self._window_nodes)]
        out: list[int] = []
        for interior_only in (True, False):
            for node in order:
                if len(out) >= num:
                    return out
                pages = node.window_pages
                if pages is None:
                    continue
                cut = len(pages) - trailing_pages if interior_only else len(pages)
                kept = list(pages)
                for j in range(max(0, cut)):
                    w = kept[j]
                    if w is not None and w not in self._window_pins and len(out) < num:
                        out.append(w)
                        kept[j] = None
                self._set_window_pages(node, kept)
        return out

    def evict_host_windows(self, num: int) -> list[int]:
        """Drop up to ``num`` host window pages, LRU nodes first."""
        out: list[int] = []
        for _, _, node in sorted((n.last_access, n.id, n) for n in self._host_window_nodes):
            if len(out) >= num:
                break
            self._drop_host_windows(node, out)
        return out

    # ===== Eviction =====

    def _device_evictable(self, node: Node) -> bool:
        return node.lock_ref == 0 and (node.h2d_op is None or node.h2d_op.launched)

    def evict_device(self, num_pages: int) -> Evicted:
        """Free at least ``num_pages`` device pages if possible, LRU first.

        Backed-up nodes are demoted to host-only; others are deleted. Device
        states of evicted nodes are freed with their pages.
        """
        heap = [
            (n.last_access, n.id, n)
            for n in self._device_leaves
            if self._device_evictable(n)
        ]
        heapq.heapify(heap)
        freed: list[int] = []
        states: list[int] = []
        windows: list[int] = []
        while heap and len(freed) < num_pages:
            _, _, node = heapq.heappop(heap)
            if node not in self._device_leaves or not self._device_evictable(node):
                continue
            for op in (node.d2h_op, node.h2d_op):
                if op is not None:
                    op.wait()
            node.d2h_op = node.h2d_op = None
            freed.extend(node.device_pages)
            self._drop_state(node, states)
            self._drop_windows(node, windows)
            parent = node.parent
            self._set_device_pages(node, None)
            if node.host_pages is None:
                # Host residency is prefix-closed, so an unbacked node has no
                # host-only children left behind.
                self._remove_leaf(node)
            if parent in self._device_leaves and self._device_evictable(parent):
                heapq.heappush(heap, (parent.last_access, parent.id, parent))
        return Evicted(pages=freed, states=states, windows=windows)

    def evict_host(self, num_pages: int) -> Evicted:
        """Free at least ``num_pages`` host pages if possible, LRU first."""
        heap = [(n.last_access, n.id, n) for n in self._host_leaves]
        heapq.heapify(heap)
        freed: list[int] = []
        states: list[int] = []
        windows: list[int] = []
        while heap and len(freed) < num_pages:
            _, _, node = heapq.heappop(heap)
            if node not in self._host_leaves:
                continue
            if node.d2h_op is not None:
                node.d2h_op.wait()
            freed.extend(node.host_pages)
            self._drop_host_state(node, states)
            self._drop_host_windows(node, windows)
            parent = node.parent
            self._set_host_pages(node, None)
            self._remove_leaf(node)
            if parent in self._host_leaves:
                heapq.heappush(heap, (parent.last_access, parent.id, parent))
        return Evicted(pages=freed, states=states, windows=windows)

    def drop_all(self) -> tuple[Evicted, Evicted]:
        """Remove every node; returns what the device and host tiers held.
        Needs no locks."""
        assert not self._window_pins, "dropping the index while leases pin windows"
        device, host, stack = [], [], list(self.root.children.values())
        states, host_states = [], []
        windows, host_windows = [], []
        while stack:
            node = stack.pop()
            assert node.lock_ref == 0
            for op in (node.d2h_op, node.h2d_op):
                if op is not None:
                    op.wait()
            device.extend(node.device_pages or ())
            host.extend(node.host_pages or ())
            self._drop_state(node, states)
            self._drop_host_state(node, host_states)
            self._drop_windows(node, windows)
            self._drop_host_windows(node, host_windows)
            stack.extend(node.children.values())
        self.__init__(page_size=self.page_size)
        return (
            Evicted(pages=device, states=states, windows=windows),
            Evicted(pages=host, states=host_states, windows=host_windows),
        )
