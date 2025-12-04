"""
DB-backed rewrite sketch of make_exit_feat_v3_2.py.

Reads TokenInfo/PairEvent from Django ORM instead of CSV and returns a list of
dicts ready for bulk insert into ExitProcessedData/ExitProcessedDataInstance.
This is intentionally standalone (not wired to adapters) so you can validate
logic before integrating it into modules/preprocessor/exit_instance.py.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, getcontext
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure adequate precision for on-chain numbers
getcontext().prec = 28


# Django bootstrap -----------------------------------------------------------
def setup_django(settings_module: str = "config.settings") -> None:
    if "DJANGO_SETTINGS_MODULE" not in os.environ:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()


# Helpers --------------------------------------------------------------------
def to_decimal(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return Decimal(0)
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            return Decimal(text)
        except InvalidOperation:
            return None
    return None


def log1p_positive(value: Optional[Decimal]) -> float:
    if value is None or value <= 0:
        return 0.0
    return float((value + 1).ln())


def signed_log1p(value: Optional[Decimal]) -> float:
    if value is None or value == 0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return float((abs(value) + 1).ln()) * sign


def safe_div(numer: Optional[Decimal], denom: Optional[Decimal]) -> Optional[Decimal]:
    if numer is None or denom in (None, Decimal(0)):
        return None
    try:
        return numer / denom
    except InvalidOperation:
        return None


# Data shapes ----------------------------------------------------------------
@dataclass
class PairEventRecord:
    timestamp: datetime
    block_number: Optional[int]
    tx_hash: Optional[str]
    evt_idx: Optional[int]
    evt_type: str
    lp_total_supply: Optional[Decimal]
    evt_log: Dict[str, object]


class ExitInstancePreprocessor:
    def __init__(self):
        from api.models import PairEvent

        self.PairEvent = PairEvent

    def _determine_base_is_token0(self, pair_idx: Optional[int]) -> bool:
        if pair_idx in (0, 1):
            return pair_idx == 0
        return True

    def _fetch_records(self, token_info) -> List[PairEventRecord]:
        qs = (
            self.PairEvent.objects.filter(token_info=token_info)
            .order_by("timestamp", "evt_idx")
            .only(
                "timestamp",
                "block_number",
                "tx_hash",
                "evt_idx",
                "evt_type",
                "lp_total_supply",
                "evt_log",
            )
        )
        return [
            PairEventRecord(
                timestamp=evt.timestamp,
                block_number=evt.block_number,
                tx_hash=evt.tx_hash,
                evt_idx=evt.evt_idx,
                evt_type=(evt.evt_type or "").lower(),
                lp_total_supply=to_decimal(evt.lp_total_supply),
                evt_log=evt.evt_log or {},
            )
            for evt in qs
        ]

    def _group_by_tx(self, records: Sequence[PairEventRecord]) -> List[List[PairEventRecord]]:
        grouped: List[List[PairEventRecord]] = []
        current: List[PairEventRecord] = []
        current_key: Optional[Tuple[datetime, Optional[int], Optional[str]]] = None
        for record in records:
            key = (record.timestamp, record.block_number, record.tx_hash)
            if current_key is None or key != current_key:
                if current:
                    grouped.append(current)
                current = [record]
                current_key = key
            else:
                current.append(record)
        if current:
            grouped.append(current)
        return grouped

    def compute_exit_features(self, token_info) -> List[Dict[str, object]]:
        """
        Build per-event features from DB events for a single token.
        Returns list of dicts keyed by ExitProcessedData fields (log1p scaled).
        """
        records = self._fetch_records(token_info)
        if not records:
            return []

        base_is_token0 = self._determine_base_is_token0(getattr(token_info, "pair_idx", None))

        rows: List[Dict[str, object]] = []
        prev_ts: Optional[datetime] = None
        prev_lp: Optional[Decimal] = None
        prev_base: Optional[Decimal] = None
        prev_quote: Optional[Decimal] = None
        prev_price: Optional[Decimal] = None
        last_mint_ts: Optional[datetime] = None

        rq_series: List[Optional[Decimal]] = []
        swap_series: List[int] = []
        mint_series: List[int] = []
        burn_series: List[int] = []
        grouped = self._group_by_tx(records)

        for local_id, group in enumerate(grouped):
            first = group[0]
            timestamp = first.timestamp.astimezone(timezone.utc)
            delta_t = Decimal(0)
            if prev_ts is not None:
                seconds = max(int((timestamp - prev_ts).total_seconds()), 0)
                delta_t = Decimal(seconds)
            prev_ts = timestamp

            quote_before, base_before, price_before, lp_before = prev_quote, prev_base, prev_price, prev_lp
            lp_current, base_current, quote_current, price_current = prev_lp, prev_base, prev_quote, prev_price
            lp_minted_amount = Decimal(0)
            lp_burned_amount = Decimal(0)
            swap_base_in_total = Decimal(0)
            swap_base_out_total = Decimal(0)
            swap_quote_in_total = Decimal(0)
            swap_quote_out_total = Decimal(0)

            is_mint = False
            is_burn = False
            is_swap = False

            for record in group:
                evt_type = record.evt_type
                payload = record.evt_log or {}
                swap_base_in = Decimal(0)
                swap_base_out = Decimal(0)
                swap_quote_in = Decimal(0)
                swap_quote_out = Decimal(0)

                if evt_type == "mint":
                    is_mint = True
                elif evt_type == "burn":
                    is_burn = True
                elif evt_type == "swap":
                    is_swap = True

                if evt_type == "swap":
                    amount0_in = to_decimal(payload.get("amount0In"))
                    amount0_out = to_decimal(payload.get("amount0Out"))
                    amount1_in = to_decimal(payload.get("amount1In"))
                    amount1_out = to_decimal(payload.get("amount1Out"))
                    if base_is_token0:
                        swap_base_in = amount0_in or Decimal(0)
                        swap_base_out = amount0_out or Decimal(0)
                        swap_quote_in = amount1_in or Decimal(0)
                        swap_quote_out = amount1_out or Decimal(0)
                    else:
                        swap_base_in = amount1_in or Decimal(0)
                        swap_base_out = amount1_out or Decimal(0)
                        swap_quote_in = amount0_in or Decimal(0)
                        swap_quote_out = amount0_out or Decimal(0)
                elif evt_type == "sync":
                    reserve0 = to_decimal(payload.get("reserve0"))
                    reserve1 = to_decimal(payload.get("reserve1"))
                    if base_is_token0:
                        base_current = reserve0
                        quote_current = reserve1
                    else:
                        base_current = reserve1
                        quote_current = reserve0
                    if reserve0 not in (None, Decimal(0)) and reserve1 not in (None, Decimal(0)):
                        if base_is_token0:
                            price_current = safe_div(reserve0, reserve1)
                        else:
                            price_current = safe_div(reserve1, reserve0)

                swap_base_in_total += swap_base_in
                swap_base_out_total += swap_base_out
                swap_quote_in_total += swap_quote_in
                swap_quote_out_total += swap_quote_out

                lp_after = record.lp_total_supply if record.lp_total_supply is not None else lp_current
                if lp_before is not None and lp_after is not None:
                    delta_lp = lp_after - lp_before
                    if delta_lp > 0:
                        lp_minted_amount += delta_lp
                    elif delta_lp < 0:
                        lp_burned_amount += -delta_lp
                lp_current = lp_after

            quote_drop_frac = Decimal(0)
            base_drop_frac = Decimal(0)
            quote_delta = Decimal(0)
            base_delta = Decimal(0)
            if quote_before is not None and quote_current is not None:
                quote_delta = quote_current - quote_before
                if quote_before > 0:
                    drop = (quote_before - quote_current) / quote_before
                    if drop > 0:
                        quote_drop_frac = drop
            if base_before is not None and base_current is not None:
                base_delta = base_current - base_before
                if base_before > 0:
                    drop = (base_before - base_current) / base_before
                    if drop > 0:
                        base_drop_frac = drop

            price_ratio_delta: Optional[Decimal] = None
            if price_current is not None and price_before is not None:
                price_ratio_delta = price_current - price_before
            elif price_current is not None:
                price_ratio_delta = price_current

            time_since_last_mint: Optional[Decimal] = None
            if last_mint_ts is not None:
                delta_sec = max(int((timestamp - last_mint_ts).total_seconds()), 0)
                time_since_last_mint = Decimal(delta_sec)
            if is_mint:
                time_since_last_mint = Decimal(0)
                last_mint_ts = timestamp

            divisor = delta_t if delta_t > 0 else None

            def per_sec(val: Optional[Decimal]) -> Optional[Decimal]:
                if val is None:
                    return None
                if divisor in (None, Decimal(0)):
                    return val
                return val / divisor

            lp_delta_per_sec = per_sec(lp_current - lp_before if lp_current is not None and lp_before is not None else None)
            quote_delta_per_sec = per_sec(quote_delta)
            lp_minted_per_sec = per_sec(lp_minted_amount)
            lp_burned_per_sec = per_sec(lp_burned_amount)

            row = {
                "local_id": local_id,
                "token_addr_idx": token_info.id,
                "event_time": timestamp,
                "tx_hash": first.tx_hash,
                "delta_t_sec": log1p_positive(delta_t),
                "is_mint_event": bool(is_mint),
                "is_burn_event": bool(is_burn),
                "is_swap_event": bool(is_swap),
                "lp_total_supply": log1p_positive(lp_current),
                "reserve_base_drop_frac": float(base_drop_frac),
                "reserve_quote": log1p_positive(quote_current),
                "reserve_quote_drop_frac": float(quote_drop_frac),
                "price_ratio": log1p_positive(price_current),
                "time_since_last_mint_sec": log1p_positive(time_since_last_mint),
                "price_ratio_delta_per_sec": signed_log1p(per_sec(price_ratio_delta)),
                "lp_supply_delta_per_sec": signed_log1p(lp_delta_per_sec),
                "reserve_quote_delta_per_sec": signed_log1p(quote_delta_per_sec),
                "lp_minted_amount_per_sec": log1p_positive(lp_minted_per_sec),
                "lp_burned_amount_per_sec": log1p_positive(lp_burned_per_sec),
                "swap_base_in_amount_per_sec": log1p_positive(per_sec(swap_base_in_total)),
                "swap_base_out_amount_per_sec": log1p_positive(per_sec(swap_base_out_total)),
                "swap_quote_in_amount_per_sec": log1p_positive(per_sec(swap_quote_in_total)),
                "swap_quote_out_amount_per_sec": log1p_positive(per_sec(swap_quote_out_total)),
            }
            rows.append(row)

            prev_lp = lp_current
            prev_base = base_current
            prev_quote = quote_current
            prev_price = price_current if price_current is not None else prev_price

            rq_series.append(quote_current)
            swap_series.append(1 if is_swap else 0)
            mint_series.append(1 if is_mint else 0)
            burn_series.append(1 if is_burn else 0)

        # Token-level features appended to each row
        rq_values = [float(x) for x in rq_series if x is not None]
        drawdown_val = 0.0
        if rq_values:
            max_rq = max(rq_values)
            final_rq = next((float(x) for x in reversed(rq_series) if x is not None), None)
            if max_rq > 0 and final_rq is not None:
                drawdown_val = max(0.0, min(1.0, 1.0 - (final_rq / max_rq)))

        def last_n_ratio(series: List[int], n: int) -> Optional[float]:
            tail = series[-n:]
            if not tail:
                return None
            return sum(tail) / len(tail)

        recent_mint10 = last_n_ratio(mint_series, 10)
        recent_mint20 = last_n_ratio(mint_series, 20)
        recent_burn10 = last_n_ratio(burn_series, 10)
        recent_burn20 = last_n_ratio(burn_series, 20)

        for row in rows:
            row["recent_mint_ratio_last10"] = recent_mint10
            row["recent_mint_ratio_last20"] = recent_mint20
            row["recent_burn_ratio_last10"] = recent_burn10
            row["recent_burn_ratio_last20"] = recent_burn20
            row["reserve_quote_drawdown"] = drawdown_val

        return rows


    def save_to_db(self, token_info, rows: List[Dict[str, object]]) -> int:
        """Persist computed rows into ExitProcessedDataInstance via bulk_create."""
        from api.models import ExitProcessedDataInstance

        if not rows:
            return 0

        objects = []
        for row in rows:
            objects.append(
                ExitProcessedDataInstance(
                    token_info=token_info,
                    event_time=row["event_time"],
                    tx_hash=row.get("tx_hash") or "",
                    delta_t_sec=row.get("delta_t_sec"),
                    is_swap_event=bool(row.get("is_swap_event")),
                    lp_total_supply=Decimal(str(row["lp_total_supply"])) if row.get("lp_total_supply") is not None else None,
                    reserve_base_drop_frac=row.get("reserve_base_drop_frac"),
                    reserve_quote=Decimal(str(row["reserve_quote"])) if row.get("reserve_quote") is not None else None,
                    reserve_quote_drop_frac=row.get("reserve_quote_drop_frac"),
                    price_ratio=row.get("price_ratio"),
                    time_since_last_mint_sec=row.get("time_since_last_mint_sec"),
                    lp_minted_amount_per_sec=Decimal(str(row["lp_minted_amount_per_sec"])) if row.get("lp_minted_amount_per_sec") is not None else None,
                    lp_burned_amount_per_sec=Decimal(str(row["lp_burned_amount_per_sec"])) if row.get("lp_burned_amount_per_sec") is not None else None,
                    recent_mint_ratio_last10=row.get("recent_mint_ratio_last10"),
                    recent_mint_ratio_last20=row.get("recent_mint_ratio_last20"),
                    recent_burn_ratio_last10=row.get("recent_burn_ratio_last10"),
                    recent_burn_ratio_last20=row.get("recent_burn_ratio_last20"),
                    reserve_quote_drawdown=row.get("reserve_quote_drawdown"),
                )
            )

        ExitProcessedDataInstance.objects.bulk_create(objects, batch_size=1000)
        return len(objects)


if __name__ == "__main__":
    raise SystemExit("Run this via exit_test.py or integrate via PreprocessorAdapter.")
