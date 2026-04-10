#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
DEFAULT_QUERIES = [
    "мигранты в россии",
    "миграция в россии",
    "нелегальные мигранты россия",
    "отношение к мигрантам в россии",
    "преступления мигрантов россия",
    "мигранты москва мнение",
]
DEFAULT_SEARCH_ORDERS = ["relevance", "date"]
DEFAULT_MAX_VIDEOS_PER_QUERY = 200
DEFAULT_MAX_COMMENTS_PER_VIDEO = 500
DEFAULT_DAILY_QUOTA_LIMIT = 10000
DEFAULT_QUOTA_RESERVE = 150
DEFAULT_SEARCH_QUOTA_SHARE = 0.35
DEFAULT_MAX_SEARCH_PAGES_PER_QUERY = 4
REQUEST_TIMEOUT = 30
API_QUOTA_COSTS = {
    "search": 100,
    "videos": 1,
    "commentThreads": 1,
}


class YouTubeApiError(RuntimeError):
    pass


@dataclass
class VideoRecord:
    video_id: str
    title: str
    channel_title: str
    published_at: str
    description: str
    queries: str
    comment_count: Optional[int]
    view_count: Optional[int]
    like_count: Optional[int]
    url: str


@dataclass
class SearchCursor:
    query: str
    order: str
    page_token: Optional[str] = None
    pages_fetched: int = 0
    exhausted: bool = False


@dataclass
class CommentCursor:
    video: VideoRecord
    downloaded_count: int = 0
    page_token: Optional[str] = None
    page_requests: int = 0
    exhausted: bool = False


@dataclass
class QuotaTracker:
    daily_limit: int = DEFAULT_DAILY_QUOTA_LIMIT
    reserve: int = DEFAULT_QUOTA_RESERVE
    units_used: int = 0
    endpoint_units: Dict[str, int] = field(default_factory=dict)
    endpoint_requests: Dict[str, int] = field(default_factory=dict)

    @property
    def usable_limit(self) -> int:
        return max(self.daily_limit - self.reserve, 0)

    def remaining_total(self) -> int:
        return max(self.daily_limit - self.units_used, 0)

    def remaining_usable(self) -> int:
        return max(self.usable_limit - self.units_used, 0)

    def can_afford(self, endpoint: str) -> bool:
        return self.remaining_usable() >= API_QUOTA_COSTS.get(endpoint, 1)

    def ensure_capacity(self, endpoint: str) -> None:
        cost = API_QUOTA_COSTS.get(endpoint, 1)
        if self.units_used + cost > self.usable_limit:
            raise YouTubeApiError(
                f"Оценочный лимит квоты исчерпан: нужно {cost}, осталось "
                f"{self.remaining_usable()} до резервного порога."
            )

    def register(self, endpoint: str) -> None:
        cost = API_QUOTA_COSTS.get(endpoint, 1)
        self.units_used += cost
        self.endpoint_units[endpoint] = self.endpoint_units.get(endpoint, 0) + cost
        self.endpoint_requests[endpoint] = self.endpoint_requests.get(endpoint, 0) + 1

    def as_dict(self) -> Dict[str, object]:
        return {
            "daily_limit": self.daily_limit,
            "reserve": self.reserve,
            "usable_limit": self.usable_limit,
            "units_used_estimate": self.units_used,
            "units_remaining_estimate": self.remaining_total(),
            "units_remaining_before_reserve_estimate": self.remaining_usable(),
            "endpoint_units_estimate": self.endpoint_units,
            "endpoint_request_count": self.endpoint_requests,
            "cost_reference": API_QUOTA_COSTS,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собирает YouTube-комментарии по тематике миграции в России."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help="YouTube Data API key. По умолчанию берется из YOUTUBE_API_KEY.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Поисковый запрос. Можно указать несколько раз.",
    )
    parser.add_argument(
        "--query-file",
        default=None,
        help="Файл со списком запросов, по одному на строку.",
    )
    parser.add_argument(
        "--published-after",
        default=None,
        help="Ограничение по дате публикации роликов в формате ISO 8601.",
    )
    parser.add_argument(
        "--published-before",
        default=None,
        help="Верхняя граница публикации роликов в формате ISO 8601.",
    )
    parser.add_argument(
        "--region-code",
        default="RU",
        help="Код региона для поиска видео. По умолчанию RU.",
    )
    parser.add_argument(
        "--relevance-language",
        default="ru",
        help="Язык релевантности в поиске. По умолчанию ru.",
    )
    parser.add_argument(
        "--search-order",
        action="append",
        dest="search_orders",
        choices=["date", "rating", "relevance", "title", "videoCount", "viewCount"],
        help="Порядок поиска. Можно указать несколько раз. По умолчанию relevance и date.",
    )
    parser.add_argument(
        "--max-videos-per-query",
        type=int,
        default=DEFAULT_MAX_VIDEOS_PER_QUERY,
        help="Максимум уникальных видео на один запрос.",
    )
    parser.add_argument(
        "--max-search-pages-per-query",
        type=int,
        default=DEFAULT_MAX_SEARCH_PAGES_PER_QUERY,
        help="Максимум страниц search.list на комбинацию запрос+order.",
    )
    parser.add_argument(
        "--max-comments-per-video",
        type=int,
        default=DEFAULT_MAX_COMMENTS_PER_VIDEO,
        help="Максимум комментариев верхнего уровня на видео.",
    )
    parser.add_argument(
        "--comment-order",
        choices=["relevance", "time"],
        default="time",
        help="Порядок скачивания комментариев. По умолчанию time.",
    )
    parser.add_argument(
        "--include-replies",
        action="store_true",
        help="Сохранять ответы на комментарии, если они есть в выдаче API.",
    )
    parser.add_argument(
        "--search-quota-share",
        type=float,
        default=DEFAULT_SEARCH_QUOTA_SHARE,
        help="Доля доступной квоты, которую можно потратить на поиск видео.",
    )
    parser.add_argument(
        "--daily-quota-limit",
        type=int,
        default=DEFAULT_DAILY_QUOTA_LIMIT,
        help="Оценочный дневной лимит квоты YouTube API. По умолчанию 10000.",
    )
    parser.add_argument(
        "--quota-reserve",
        type=int,
        default=DEFAULT_QUOTA_RESERVE,
        help="Сколько units оставить в резерве и не тратить.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Пауза между API-запросами в секундах.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Папка для итоговых CSV/JSON-файлов.",
    )
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "youtube-migration-parser/2.0",
        }
    )
    return session


def load_queries(args: argparse.Namespace) -> List[str]:
    queries: List[str] = []
    if args.query_file:
        query_path = Path(args.query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Файл с запросами не найден: {query_path}")
        with query_path.open("r", encoding="utf-8") as file:
            for line in file:
                query = line.strip()
                if query:
                    queries.append(query)

    if args.queries:
        queries.extend(args.queries)

    if not queries:
        queries = list(DEFAULT_QUERIES)

    # Сохраняем порядок и убираем повторы.
    return list(dict.fromkeys(queries))


def api_get(
    session: requests.Session,
    api_key: str,
    endpoint: str,
    params: Dict[str, object],
    quota_tracker: QuotaTracker,
    sleep_seconds: float = 0.0,
) -> Dict[str, object]:
    quota_tracker.ensure_capacity(endpoint)
    response = session.get(
        f"{API_BASE}/{endpoint}",
        params={**params, "key": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    quota_tracker.register(endpoint)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    if not response.ok:
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        raise YouTubeApiError(f"API error for {endpoint}: {payload}")
    return response.json()


def chunked(values: List[str], size: int) -> Iterator[List[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def parse_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def search_videos_page(
    session: requests.Session,
    api_key: str,
    cursor: SearchCursor,
    region_code: str,
    relevance_language: str,
    published_after: Optional[str],
    published_before: Optional[str],
    quota_tracker: QuotaTracker,
    sleep_seconds: float,
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "part": "snippet",
        "q": cursor.query,
        "type": "video",
        "maxResults": 50,
        "order": cursor.order,
        "regionCode": region_code,
        "relevanceLanguage": relevance_language,
    }
    if cursor.page_token:
        params["pageToken"] = cursor.page_token
    if published_after:
        params["publishedAfter"] = published_after
    if published_before:
        params["publishedBefore"] = published_before

    payload = api_get(session, api_key, "search", params, quota_tracker, sleep_seconds)
    items = payload.get("items", [])
    return {
        "video_ids": [
            item.get("id", {}).get("videoId")
            for item in items
            if item.get("id", {}).get("videoId")
        ],
        "next_page_token": payload.get("nextPageToken"),
    }


def fetch_video_details(
    session: requests.Session,
    api_key: str,
    video_ids: List[str],
    query_map: Dict[str, List[str]],
    quota_tracker: QuotaTracker,
    sleep_seconds: float,
) -> List[VideoRecord]:
    videos: List[VideoRecord] = []
    for group in chunked(video_ids, 50):
        payload = api_get(
            session,
            api_key,
            "videos",
            {
                "part": "snippet,statistics",
                "id": ",".join(group),
                "maxResults": len(group),
            },
            quota_tracker,
            sleep_seconds,
        )
        for item in payload.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            videos.append(
                VideoRecord(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    description=snippet.get("description", ""),
                    queries=" | ".join(query_map.get(item["id"], [])),
                    comment_count=parse_int(statistics.get("commentCount")),
                    view_count=parse_int(statistics.get("viewCount")),
                    like_count=parse_int(statistics.get("likeCount")),
                    url=f"https://www.youtube.com/watch?v={item['id']}",
                )
            )
    return videos


def fetch_comment_threads_page(
    session: requests.Session,
    api_key: str,
    cursor: CommentCursor,
    comment_order: str,
    include_replies: bool,
    per_video_limit: int,
    quota_tracker: QuotaTracker,
    sleep_seconds: float,
) -> Dict[str, object]:
    remaining_for_video = per_video_limit - cursor.downloaded_count
    if remaining_for_video <= 0:
        return {"rows": [], "next_page_token": None, "exhausted": True}

    params: Dict[str, object] = {
        "part": "snippet,replies",
        "videoId": cursor.video.video_id,
        "maxResults": min(100, remaining_for_video),
        "order": comment_order,
        "textFormat": "plainText",
    }
    if cursor.page_token:
        params["pageToken"] = cursor.page_token

    try:
        payload = api_get(
            session,
            api_key,
            "commentThreads",
            params,
            quota_tracker,
            sleep_seconds,
        )
    except YouTubeApiError as exc:
        if "commentsDisabled" in str(exc):
            return {"rows": [], "next_page_token": None, "exhausted": True}
        raise

    rows: List[Dict[str, object]] = []
    items = payload.get("items", [])
    for item in items:
        if len(rows) >= remaining_for_video:
            break
        thread_id = item.get("id", "")
        top_level_comment = item.get("snippet", {}).get("topLevelComment", {})
        top_comment = top_level_comment.get("snippet", {})
        if top_comment:
            rows.append(
                build_comment_row(
                    video=cursor.video,
                    thread_id=thread_id,
                    snippet=top_comment,
                    comment_id=str(top_level_comment.get("id", "")),
                    parent_author=None,
                    is_reply=False,
                )
            )

        if include_replies:
            for reply in item.get("replies", {}).get("comments", []):
                if len(rows) >= remaining_for_video:
                    break
                reply_snippet = reply.get("snippet", {})
                if reply_snippet:
                    rows.append(
                        build_comment_row(
                            video=cursor.video,
                            thread_id=thread_id,
                            snippet=reply_snippet,
                            comment_id=str(reply.get("id", "")),
                            parent_author=top_comment.get("authorDisplayName"),
                            is_reply=True,
                        )
                    )

    next_page_token = payload.get("nextPageToken")
    return {
        "rows": rows[:remaining_for_video],
        "next_page_token": next_page_token,
        "exhausted": not next_page_token or not items or len(rows) == 0,
    }


def build_comment_row(
    video: VideoRecord,
    thread_id: str,
    snippet: Dict[str, object],
    comment_id: str,
    parent_author: Optional[str],
    is_reply: bool,
) -> Dict[str, object]:
    text = normalize_whitespace(str(snippet.get("textDisplay", "")))
    return {
        "video_id": video.video_id,
        "video_url": video.url,
        "video_title": video.title,
        "video_channel": video.channel_title,
        "video_published_at": video.published_at,
        "video_queries": video.queries,
        "thread_id": thread_id,
        "comment_id": comment_id,
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "author_name": snippet.get("authorDisplayName", ""),
        "author_channel_url": snippet.get("authorChannelUrl", ""),
        "author_channel_id": (snippet.get("authorChannelId", {}) or {}).get("value", ""),
        "parent_author_name": parent_author or "",
        "is_reply": is_reply,
        "like_count": parse_int(snippet.get("likeCount")),
        "text": text,
        "text_length": len(text),
    }


def summarise(
    comments: List[Dict[str, object]],
    videos: List[VideoRecord],
    query_stats: Dict[str, Dict[str, int]],
) -> Dict[str, object]:
    top_videos = sorted(
        videos,
        key=lambda video: (video.comment_count or 0, video.view_count or 0),
        reverse=True,
    )[:10]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_count": len(videos),
        "comment_count": len(comments),
        "query_stats": query_stats,
        "top_videos": [
            {
                "video_id": video.video_id,
                "title": video.title,
                "channel_title": video.channel_title,
                "published_at": video.published_at,
                "comment_count": video.comment_count,
                "view_count": video.view_count,
                "queries": video.queries,
                "url": video.url,
            }
            for video in top_videos
        ],
    }


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_comments_csv(output_dir: Path, comments: List[Dict[str, object]]) -> Path:
    destination = output_dir / "youtube_comments.csv"
    fieldnames = [
        "video_id",
        "video_url",
        "video_title",
        "video_channel",
        "video_published_at",
        "video_queries",
        "thread_id",
        "comment_id",
        "published_at",
        "updated_at",
        "author_name",
        "author_channel_url",
        "author_channel_id",
        "parent_author_name",
        "is_reply",
        "like_count",
        "text",
        "text_length",
    ]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comments)
    return destination


def write_videos_csv(output_dir: Path, videos: List[VideoRecord]) -> Path:
    destination = output_dir / "youtube_videos.csv"
    fieldnames = [
        "video_id",
        "title",
        "channel_title",
        "published_at",
        "description",
        "queries",
        "comment_count",
        "view_count",
        "like_count",
        "url",
    ]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for video in videos:
            writer.writerow(asdict(video))
    return destination


def write_video_stats_csv(output_dir: Path, rows: List[Dict[str, object]]) -> Path:
    destination = output_dir / "video_comment_stats.csv"
    fieldnames = [
        "video_id",
        "video_url",
        "video_title",
        "video_channel",
        "video_published_at",
        "video_queries",
        "api_reported_comment_count",
        "max_comments_requested",
        "estimated_downloadable_in_run",
        "downloaded_comment_count",
        "comment_page_requests",
        "download_coverage_vs_api_reported",
        "download_coverage_vs_run_limit",
    ]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_summary_json(output_dir: Path, summary: Dict[str, object]) -> Path:
    destination = output_dir / "summary.json"
    with destination.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return destination


def print_progress(message: str) -> None:
    print(message, file=sys.stderr)


def build_video_comment_stats(
    video: VideoRecord,
    downloaded_count: int,
    page_requests: int,
    max_comments: int,
) -> Dict[str, object]:
    reported_total = video.comment_count or 0
    estimated_downloadable = min(reported_total, max_comments) if reported_total else 0
    return {
        "video_id": video.video_id,
        "video_url": video.url,
        "video_title": video.title,
        "video_channel": video.channel_title,
        "video_published_at": video.published_at,
        "video_queries": video.queries,
        "api_reported_comment_count": reported_total,
        "max_comments_requested": max_comments,
        "estimated_downloadable_in_run": estimated_downloadable,
        "downloaded_comment_count": downloaded_count,
        "comment_page_requests": page_requests,
        "download_coverage_vs_api_reported": (
            round(downloaded_count / reported_total, 4) if reported_total else 0.0
        ),
        "download_coverage_vs_run_limit": (
            round(downloaded_count / estimated_downloadable, 4)
            if estimated_downloadable
            else 0.0
        ),
    }


def search_budget_limit(quota_tracker: QuotaTracker, share: float, query_count: int) -> int:
    usable_limit = quota_tracker.usable_limit
    share = min(max(share, 0.05), 0.95)
    share_budget = int(usable_limit * share)
    minimum_budget = min(query_count * API_QUOTA_COSTS["search"], usable_limit)
    return min(max(share_budget, minimum_budget), usable_limit)


def collect_video_ids(
    session: requests.Session,
    api_key: str,
    queries: List[str],
    search_orders: List[str],
    max_videos_per_query: int,
    max_search_pages_per_query: int,
    region_code: str,
    relevance_language: str,
    published_after: Optional[str],
    published_before: Optional[str],
    quota_tracker: QuotaTracker,
    search_quota_share: float,
    sleep_seconds: float,
) -> Dict[str, object]:
    query_unique_ids: Dict[str, List[str]] = {query: [] for query in queries}
    query_seen_ids: Dict[str, set[str]] = {query: set() for query in queries}
    query_stats: Dict[str, Dict[str, int]] = {
        query: {
            "search_pages": 0,
            "unique_videos": 0,
            "duplicate_hits": 0,
        }
        for query in queries
    }
    video_query_map: Dict[str, List[str]] = {}
    all_video_ids: List[str] = []
    duplicate_video_hits = 0
    cursors = [
        SearchCursor(query=query, order=order)
        for query in queries
        for order in search_orders
    ]

    max_search_units = search_budget_limit(quota_tracker, search_quota_share, len(queries))
    search_units_started = quota_tracker.units_used

    while True:
        progress_made = False
        for cursor in cursors:
            if cursor.exhausted:
                continue
            if cursor.pages_fetched >= max_search_pages_per_query:
                cursor.exhausted = True
                continue
            if len(query_unique_ids[cursor.query]) >= max_videos_per_query:
                cursor.exhausted = True
                continue
            if not quota_tracker.can_afford("search"):
                break
            if (quota_tracker.units_used - search_units_started) >= max_search_units:
                break

            payload = search_videos_page(
                session=session,
                api_key=api_key,
                cursor=cursor,
                region_code=region_code,
                relevance_language=relevance_language,
                published_after=published_after,
                published_before=published_before,
                quota_tracker=quota_tracker,
                sleep_seconds=sleep_seconds,
            )
            cursor.pages_fetched += 1
            query_stats[cursor.query]["search_pages"] += 1
            progress_made = True

            new_unique_for_query = 0
            for video_id in payload["video_ids"]:
                if video_id in query_seen_ids[cursor.query]:
                    query_stats[cursor.query]["duplicate_hits"] += 1
                    duplicate_video_hits += 1
                    continue

                query_seen_ids[cursor.query].add(video_id)
                if len(query_unique_ids[cursor.query]) < max_videos_per_query:
                    query_unique_ids[cursor.query].append(video_id)
                    new_unique_for_query += 1
                    if video_id not in video_query_map:
                        all_video_ids.append(video_id)
                        video_query_map[video_id] = [cursor.query]
                    elif cursor.query not in video_query_map[video_id]:
                        video_query_map[video_id].append(cursor.query)

            query_stats[cursor.query]["unique_videos"] = len(query_unique_ids[cursor.query])
            cursor.page_token = payload["next_page_token"]

            print_progress(
                "[search] "
                f"{cursor.query} | order={cursor.order} | page={cursor.pages_fetched} | "
                f"new_unique={new_unique_for_query} | total_unique={len(query_unique_ids[cursor.query])}"
            )

            if not payload["next_page_token"] or new_unique_for_query == 0:
                cursor.exhausted = True

        if not progress_made:
            break
        if not quota_tracker.can_afford("search"):
            break
        if (quota_tracker.units_used - search_units_started) >= max_search_units:
            break

    return {
        "all_video_ids": list(dict.fromkeys(all_video_ids)),
        "video_query_map": video_query_map,
        "query_stats": query_stats,
        "duplicate_video_hits": duplicate_video_hits,
        "search_budget_units": max_search_units,
    }


def download_comments_round_robin(
    session: requests.Session,
    api_key: str,
    videos: List[VideoRecord],
    max_comments_per_video: int,
    comment_order: str,
    include_replies: bool,
    quota_tracker: QuotaTracker,
    sleep_seconds: float,
) -> Dict[str, object]:
    cursors = [
        CommentCursor(video=video)
        for video in videos
        if (video.comment_count or 0) > 0
    ]
    all_comments: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    processed_ids: set[str] = set()
    round_number = 0

    while True:
        active_cursors = [
            cursor
            for cursor in cursors
            if not cursor.exhausted and cursor.downloaded_count < max_comments_per_video
        ]
        if not active_cursors or not quota_tracker.can_afford("commentThreads"):
            break

        round_number += 1
        round_downloaded = 0
        print_progress(
            f"[comments] round={round_number} active_videos={len(active_cursors)} "
            f"quota_left_before_reserve={quota_tracker.remaining_usable()}"
        )

        for cursor in active_cursors:
            if not quota_tracker.can_afford("commentThreads"):
                break
            payload = fetch_comment_threads_page(
                session=session,
                api_key=api_key,
                cursor=cursor,
                comment_order=comment_order,
                include_replies=include_replies,
                per_video_limit=max_comments_per_video,
                quota_tracker=quota_tracker,
                sleep_seconds=sleep_seconds,
            )
            cursor.page_requests += 1
            rows = payload["rows"]
            cursor.downloaded_count += len(rows)
            cursor.page_token = payload["next_page_token"]
            cursor.exhausted = payload["exhausted"] or cursor.downloaded_count >= max_comments_per_video
            all_comments.extend(rows)
            round_downloaded += len(rows)

        if round_downloaded == 0:
            break

    for cursor in cursors:
        stats_rows.append(
            build_video_comment_stats(
                video=cursor.video,
                downloaded_count=cursor.downloaded_count,
                page_requests=cursor.page_requests,
                max_comments=max_comments_per_video,
            )
        )
        processed_ids.add(cursor.video.video_id)

    for video in videos:
        if video.video_id not in processed_ids:
            stats_rows.append(
                build_video_comment_stats(
                    video=video,
                    downloaded_count=0,
                    page_requests=0,
                    max_comments=max_comments_per_video,
                )
            )

    return {
        "comments": all_comments,
        "video_comment_stats": stats_rows,
        "rounds": round_number,
    }


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print(
            "Не найден API-ключ. Укажите --api-key или задайте переменную окружения "
            "YOUTUBE_API_KEY.",
            file=sys.stderr,
        )
        return 1

    try:
        queries = load_queries(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    search_orders = args.search_orders or list(DEFAULT_SEARCH_ORDERS)
    output_dir = ensure_output_dir(args.output_dir)
    session = make_session()
    quota_tracker = QuotaTracker(
        daily_limit=args.daily_quota_limit,
        reserve=args.quota_reserve,
    )

    try:
        print_progress(
            f"[start] queries={len(queries)} | search_orders={','.join(search_orders)} | "
            f"usable_quota={quota_tracker.usable_limit}"
        )

        search_result = collect_video_ids(
            session=session,
            api_key=args.api_key,
            queries=queries,
            search_orders=search_orders,
            max_videos_per_query=args.max_videos_per_query,
            max_search_pages_per_query=args.max_search_pages_per_query,
            region_code=args.region_code,
            relevance_language=args.relevance_language,
            published_after=args.published_after,
            published_before=args.published_before,
            quota_tracker=quota_tracker,
            search_quota_share=args.search_quota_share,
            sleep_seconds=args.sleep,
        )

        unique_video_ids = search_result["all_video_ids"]
        if not unique_video_ids:
            print("По заданным запросам не найдено видео.", file=sys.stderr)
            return 1

        videos = fetch_video_details(
            session=session,
            api_key=args.api_key,
            video_ids=unique_video_ids,
            query_map=search_result["video_query_map"],
            quota_tracker=quota_tracker,
            sleep_seconds=args.sleep,
        )
        videos.sort(
            key=lambda video: (video.comment_count or 0, video.view_count or 0),
            reverse=True,
        )
        print_progress(
            f"[videos] уникальных видео={len(videos)} | "
            f"повторов пропущено={search_result['duplicate_video_hits']}"
        )

        comments_result = download_comments_round_robin(
            session=session,
            api_key=args.api_key,
            videos=videos,
            max_comments_per_video=args.max_comments_per_video,
            comment_order=args.comment_order,
            include_replies=args.include_replies,
            quota_tracker=quota_tracker,
            sleep_seconds=args.sleep,
        )

        summary = summarise(
            comments=comments_result["comments"],
            videos=videos,
            query_stats=search_result["query_stats"],
        )
        summary["search_strategy"] = {
            "search_orders": search_orders,
            "max_videos_per_query": args.max_videos_per_query,
            "max_search_pages_per_query": args.max_search_pages_per_query,
            "search_quota_share": args.search_quota_share,
            "comment_order": args.comment_order,
            "max_comments_per_video": args.max_comments_per_video,
            "comment_rounds_completed": comments_result["rounds"],
            "search_budget_units": search_result["search_budget_units"],
        }
        summary["duplicate_video_hits_skipped"] = search_result["duplicate_video_hits"]
        summary["quota_estimate"] = quota_tracker.as_dict()

        videos_path = write_videos_csv(output_dir, videos)
        comments_path = write_comments_csv(output_dir, comments_result["comments"])
        stats_path = write_video_stats_csv(output_dir, comments_result["video_comment_stats"])
        summary_path = write_summary_json(output_dir, summary)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print_progress(f"[saved] {videos_path}")
        print_progress(f"[saved] {comments_path}")
        print_progress(f"[saved] {stats_path}")
        print_progress(f"[saved] {summary_path}")
        return 0
    except KeyboardInterrupt:
        print("Остановлено пользователем.", file=sys.stderr)
        return 130
    except (requests.RequestException, YouTubeApiError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
