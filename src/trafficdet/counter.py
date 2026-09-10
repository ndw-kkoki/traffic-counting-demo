"""トラッキング結果(オブジェクトIDごとの重心軌跡)から、
指定したライン(2点)を通過した物体をクラス別にカウントする。

判定ロジック(`crossed_line`)は動画・トラッカーに依存しない純粋関数にして、
ユニットテストで検証できるようにしている。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def _side(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """点(px,py)がライン(x1,y1)-(x2,y2)のどちら側にあるかを符号で返す。"""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def crossed_line(
    prev_point: tuple[float, float],
    curr_point: tuple[float, float],
    line: tuple[float, float, float, float],
) -> bool:
    """物体の重心が prev_point -> curr_point の間にラインを横切ったか判定する。"""
    x1, y1, x2, y2 = line
    s_prev = _side(*prev_point, x1, y1, x2, y2)
    s_curr = _side(*curr_point, x1, y1, x2, y2)
    if s_prev == 0 or s_curr == 0:
        return False
    return (s_prev > 0) != (s_curr > 0)


@dataclass
class LineCounter:
    """ライン(x1,y1,x2,y2)を通過したトラックIDをクラス別に一度だけカウントする。"""

    line: tuple[float, float, float, float]
    _last_point: dict[int, tuple[float, float]] = field(default_factory=dict)
    _counted_ids: set[int] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, track_id: int, class_name: str, centroid: tuple[float, float]) -> bool:
        """1フレーム分のトラック位置を反映する。今回のフレームで新規カウントされたらTrueを返す。"""
        prev = self._last_point.get(track_id)
        self._last_point[track_id] = centroid

        if prev is None or track_id in self._counted_ids:
            return False

        if crossed_line(prev, centroid, self.line):
            self._counted_ids.add(track_id)
            self.counts[class_name] += 1
            return True
        return False

    @property
    def total(self) -> int:
        return sum(self.counts.values())
