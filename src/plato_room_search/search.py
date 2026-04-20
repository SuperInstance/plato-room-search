"""Cross-room tile search."""

from typing import Optional
from enum import Enum

class MatchType(Enum):
    EXACT = "exact"
    TAG = "tag"
    DOMAIN = "domain"
    KEYWORD = "keyword"
    FUZZY = "fuzzy"

class RoomSearcher:
    def __init__(self, rooms: dict[str, list[dict]] = None):
        self.rooms = rooms or {}

    def add_room(self, room_id: str, tiles: list[dict]):
        self.rooms[room_id] = tiles

    def remove_room(self, room_id: str):
        self.rooms.pop(room_id, None)

    def search(self, query: str, match_type: MatchType = MatchType.KEYWORD,
               room_boost: str = None, top_n: int = 10) -> list[dict]:
        q_lower = query.lower().strip()
        results = []
        for room_id, tiles in self.rooms.items():
            for t in tiles:
                score = 0.0
                content = t.get("content", "").lower()
                if match_type == MatchType.EXACT and q_lower == content:
                    score = 1.0
                elif match_type == MatchType.TAG:
                    tags = t.get("tags", [])
                    if q_lower in [tag.lower() for tag in tags]:
                        score = 0.9
                elif match_type == MatchType.DOMAIN:
                    if t.get("domain", "").lower() == q_lower:
                        score = 0.8
                elif match_type == MatchType.KEYWORD:
                    q_words = set(q_lower.split())
                    c_words = set(content.split())
                    if q_words and c_words:
                        score = len(q_words & c_words) / len(q_words | c_words)
                elif match_type == MatchType.FUZZY:
                    q_words = set(q_lower.split())
                    c_words = set(content.split())
                    overlap = q_words & c_words
                    if overlap:
                        score = len(overlap) / max(len(q_words), 1) * 0.5
                if score > 0:
                    if room_boost and room_id == room_boost:
                        score *= 1.5
                    score += t.get("confidence", 0.5) * 0.2
                    t["_score"] = score
                    t["_room"] = room_id
                    results.append(t)
        results.sort(key=lambda x: -x.get("_score", 0))
        return results[:top_n]

    def room_tile_count(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.rooms.items()}

    @property
    def stats(self) -> dict:
        return {"rooms": len(self.rooms), "total_tiles": sum(len(v) for v in self.rooms.values())}
