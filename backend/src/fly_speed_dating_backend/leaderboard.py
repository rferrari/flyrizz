"""Supabase-backed leaderboard + match history.

Uses the SECRET (service_role) key -- this module only ever runs on the
trusted backend, never in the browser, so bypassing RLS here is safe (and
required, since anon/authenticated only have read policies -- see the
`create_leaderboard_and_match_log` migration).
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

# The .env lives at the project root (sibling of backend/), not inside
# backend/ -- matches this project's existing NEUPRINT_TOKEN convention.
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SECRET_KEY"]
        _client = create_client(url, key)
    return _client


def ensure_player(name: str, role: str) -> None:
    """Create the leaderboard row on first sight of this name; a no-op if it
    already exists (matches/died persist across that name's fly generations).
    """
    c = _get_client()
    c.table("leaderboard").upsert(
        {"name": name, "role": role}, on_conflict="name", ignore_duplicates=True
    ).execute()


def record_match(name: str, role: str, accept: float, threshold: float) -> int:
    """Increment `name`'s persisted match count and log the evaluation."""
    c = _get_client()
    current = c.table("leaderboard").select("matches").eq("name", name).single().execute()
    new_total = current.data["matches"] + 1
    c.table("leaderboard").update({"matches": new_total}).eq("name", name).execute()
    _log_evaluation(name, role, accept, threshold, matched=True)
    return new_total


def record_reject(name: str, role: str, accept: float, threshold: float) -> None:
    _log_evaluation(name, role, accept, threshold, matched=False)


def _log_evaluation(name: str, role: str, accept: float, threshold: float, matched: bool) -> None:
    c = _get_client()
    c.table("match_log").insert({
        "fly_name": name, "role": role, "accept": accept,
        "match_threshold": threshold, "matched": matched,
    }).execute()


def record_death(name: str) -> None:
    c = _get_client()
    c.table("leaderboard").update({"died": True}).eq("name", name).execute()


def top(n: int = 10) -> list[dict]:
    c = _get_client()
    result = (
        c.table("leaderboard")
        .select("name,role,matches,died")
        .order("matches", desc=True)
        .limit(n)
        .execute()
    )
    return result.data
