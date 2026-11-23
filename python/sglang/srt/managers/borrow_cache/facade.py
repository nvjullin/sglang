"""What the scheduler holds as ``tree_cache`` when a cache service owns the KV.

Prefix reuse, eviction and transfers happen behind the borrow API, so the
scheduler sees a cache without prefix sharing. Operations that would move KV
ownership through the old interface fail loudly: their call sites must take
the borrow path instead.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from sglang.srt.managers.borrow_cache.client import BorrowCacheClient
from sglang.srt.mem_cache.borrow.api import CacheKey
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    LendingPrefixCache,
    EvictResult,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv.storage import SubPoolUsage
    from sglang.srt.mem_cache.radix_cache import RadixKey


class _Root:
    """Stands in for tree node handles the scheduler passes around unread."""

    __slots__ = ()


_ROOT = _Root()


class BorrowTreeFacade(LendingPrefixCache):

    def __init__(
        self,
        *,
        client: BorrowCacheClient,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        page_size: int,
        bind_pool_tables: Callable[[], None],
    ):
        self.client = client
        # Re-applies the scheduler-side table setup the service relies on
        # (see build.py); the scheduler's pool clear undoes it.
        self.bind_pool_tables = bind_pool_tables
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.logical_page_size = page_size
        self.disable = False
        self.root_node = _ROOT

    def _forbidden(self, op: str):
        raise NotImplementedError(
            f"tree_cache.{op} is not available with SGLANG_ENABLE_DECOUPLED_HICACHE; "
            "this call site must use the borrow API"
        )

    # ===== Prefix lookups: a miss, the service decides at admission =====

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        return MatchResult(
            device_prefix_len=0,
            last_device_node=_ROOT,
            last_host_node=_ROOT,
            best_match_node=_ROOT,
            window_host_hit_lengths=MappingProxyType({}),
        )

    def supports_prefix_sharing(self) -> bool:
        return False

    def touch_prefix(self, key: RadixKey) -> None:
        # The advisory query lets the service refresh the prefix's recency.
        tokens = key.token_ids if key.limit is None else key.token_ids[: key.limit]
        self.client.service.query(
            [CacheKey(token_ids=tokens, namespace=(key.extra_key, key.cache_salt))]
        )

    def lock(self, node: Any):
        return None

    def unlock(self, lock: Any) -> None:
        assert lock is None, lock

    def inc_lock_ref(self, node: Any):
        self._forbidden("inc_lock_ref")

    def dec_lock_ref(self, node: Any, params: Any = None):
        self._forbidden("dec_lock_ref")

    # ===== Ownership moves: only through the borrow API =====

    def checkpoint(self, req: Req, *, up_to: int, **kwargs):
        self._forbidden("checkpoint")

    def free_kv_row(self, kv: Any, ranges: list[tuple[int, int]]) -> None:
        self._forbidden("free_kv_row")

    def prefix_device_indices(self, req: Req):
        self._forbidden("prefix_device_indices")

    def init_load_back(self, *args, **kwargs):
        self._forbidden("init_load_back")

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()

    # ===== Control plane =====

    def reset(self):
        if not self.client.service.flush():
            raise RuntimeError("cannot flush the cache service while leases are open")

    def check_hicache_events(self) -> None:
        self.client.poll()

    def ready_to_load_host_cache(self) -> int:
        # Loads start at the batch fence (Scheduler.run_batch), not here.
        return -1

    def flush_pending_backups(self) -> None:
        pass

    def transfers_in_flight(self) -> bool:
        return False

    def release_host_resources(self) -> None:
        self.client.close()

    def pool_stats(self, *, windows: Sequence[SubPoolUsage], target_window: str):
        """Device usage as the cache service reports it. The service's window
        pages are the target's window sub-pool, named as in ``windows``."""
        from sglang.srt.managers.scheduler_components.pool_stats_observer import (
            PoolStats,
            WindowPoolStats,
        )

        st = self.client.service.stats()
        used = st.device_slots - st.device_free_slots - st.device_evictable_slots
        stats = PoolStats(
            full_num_used=used,
            full_token_usage=used / st.device_slots,
            full_available_size=st.device_free_slots,
            full_evictable_size=st.device_evictable_slots,
            full_capacity=st.device_slots,
            target_window=target_window,
        )
        if st.window_pages:
            (usage,) = (u for u in windows if u.name == target_window)
            ps = self.page_size
            cap, free = st.window_pages * ps, st.window_free_pages * ps
            evictable = st.evictable_window_pages * ps
            stats.windows = (
                WindowPoolStats(
                    name=usage.label,
                    sub_pool=usage.name,
                    capacity=cap,
                    available_size=free,
                    evictable_size=evictable,
                    num_used=cap - free - evictable,
                    token_usage=(cap - free - evictable) / cap,
                ),
            )
        if st.state_slots:
            # Every cached state is evictable: none is pinned between calls.
            held = st.state_slots - st.state_free_slots - st.cached_states
            stats.is_hybrid_ssm = True
            stats.mamba_capacity = st.state_slots
            stats.mamba_num_used = held
            stats.mamba_usage = held / st.state_slots
            stats.mamba_available_size = st.state_free_slots
            stats.mamba_evictable_size = st.cached_states
        return stats

    def host_tier_usage(self) -> tuple[int, int]:
        """(used, total) host-tier tokens as the cache service reports them."""
        st = self.client.service.stats()
        return st.host_slots - st.host_free_slots, st.host_slots

    def host_state_usage(self) -> Optional[tuple[int, int]]:
        """(used, total) host-tier recurrent states; None without them."""
        st = self.client.service.stats()
        if not st.host_state_slots:
            return None
        return st.host_state_slots - st.host_state_free_slots, st.host_state_slots

    def available_and_evictable_str(self) -> str:
        st = self.client.service.stats()
        return (
            f"cache service: free={st.device_free_slots}, "
            f"evictable={st.device_evictable_slots}, total={st.device_slots}, "
            f"leases={st.open_leases}"
        )

    def pretty_print(self) -> None:
        pass

    def total_size(self) -> int:
        return 0
