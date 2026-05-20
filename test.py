from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import trafilatura
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# Path("News/classified_links.parquet")

INPUT_PATH = "https://baodautu.vn/tap-doan-danh-khoi-trinh-ke-hoach-tang-von-len-gan-1400-ty-dong-d164466.html"
OUTPUT_PATH = Path("News/article_contents.parquet")
CHECKPOINT_DIR = Path("News/article_content_parts")

LINK_COLUMN = "link"
ORIGINAL_INDEX_COLUMN = "original_index"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

_THREAD_LOCAL = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_THREAD_LOCAL, "session"):
        session = requests.Session()

        retry = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=64,
            pool_maxsize=64,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL.session = session

    return _THREAD_LOCAL.session


def extract_article_text(html: str) -> str:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return text or ""


def fetch_article(url: str, timeout: int = 15) -> dict:
    record = {
        LINK_COLUMN: url,
        "content": "",
        "content_length": 0,
        "status_code": pd.NA,
        "success": False,
        "error": "",
    }

    try:
        session = get_session()
        response = session.get(url, headers=HEADERS, timeout=timeout)
        status_code = response.status_code
        response.raise_for_status()

        if response.apparent_encoding:
            response.encoding = response.apparent_encoding

        content = extract_article_text(response.text)

        record.update(
            {
                "content": content,
                "content_length": len(content),
                "status_code": status_code,
                "success": bool(content),
            }
        )

        if not content:
            record["error"] = "Empty extracted content"

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def load_links(
    input_path: Path = INPUT_PATH,
    link_column: str = LINK_COLUMN,
    original_index_column: str = ORIGINAL_INDEX_COLUMN,
) -> pd.DataFrame:
    links = pd.read_parquet(
        input_path,
        columns=[original_index_column, link_column],
    )

    links[link_column] = links[link_column].astype("string").str.strip()
    links = links.loc[
        links[link_column].notna() & links[link_column].ne("")
    ].copy()

    links = links.sort_values(
        original_index_column,
        kind="mergesort",
    ).reset_index(drop=True)

    return links


def load_successful_links(checkpoint_dir: Path) -> set[str]:
    if not checkpoint_dir.exists():
        return set()

    successful_links = set()

    for path in checkpoint_dir.glob("part_*.parquet"):
        try:
            part = pd.read_parquet(path, columns=[LINK_COLUMN, "success"])
            success_mask = part["success"].fillna(False).astype(bool)
            successful_links.update(part.loc[success_mask, LINK_COLUMN].dropna().astype(str))
        except Exception:
            pass

    return successful_links


def get_next_batch_id(checkpoint_dir: Path) -> int:
    existing_batch_ids = []

    for path in checkpoint_dir.glob("part_*.parquet"):
        try:
            existing_batch_ids.append(int(path.stem.removeprefix("part_")))
        except ValueError:
            pass

    if not existing_batch_ids:
        return 0
    return max(existing_batch_ids) + 1


def save_batch(records: list[dict], checkpoint_dir: Path, batch_id: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    output_file = checkpoint_dir / f"part_{batch_id:06d}.parquet"
    temp_file = checkpoint_dir / f"part_{batch_id:06d}.tmp.parquet"

    pd.DataFrame(records).to_parquet(temp_file, index=False)
    temp_file.replace(output_file)


def crawl_unique_links(
    links: pd.Series,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    timeout: int = 15,
    max_workers: int = 32,
    batch_size: int = 1000,
) -> pd.DataFrame:
    unique_links = links.drop_duplicates().astype(str).tolist()

    successful_links = load_successful_links(checkpoint_dir)
    pending_links = [url for url in unique_links if url not in successful_links]

    print(f"Unique links: {len(unique_links):,}")
    print(f"Already successful: {len(successful_links):,}")
    print(f"Pending: {len(pending_links):,}")

    if pending_links:
        next_batch_id = get_next_batch_id(checkpoint_dir)
        batch = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_article, url, timeout): url
                for url in pending_links
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Crawling",
            ):
                batch.append(future.result())

                if len(batch) >= batch_size:
                    save_batch(batch, checkpoint_dir, next_batch_id)
                    next_batch_id += 1
                    batch.clear()

        if batch:
            save_batch(batch, checkpoint_dir, next_batch_id)

    part_files = sorted(checkpoint_dir.glob("part_*.parquet"))

    if not part_files:
        return pd.DataFrame(
            columns=[
                LINK_COLUMN,
                "content",
                "content_length",
                "status_code",
                "success",
                "error",
            ]
        )

    return pd.concat(
        [pd.read_parquet(path) for path in part_files],
        ignore_index=True,
    ).drop_duplicates(subset=[LINK_COLUMN], keep="last")


def build_article_content_dataset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    link_column: str = LINK_COLUMN,
    original_index_column: str = ORIGINAL_INDEX_COLUMN,
    timeout: int = 15,
    max_workers: int = 32,
    batch_size: int = 1000,
) -> pd.DataFrame:
    links_with_index = load_links(
        input_path=input_path,
        link_column=link_column,
        original_index_column=original_index_column,
    )

    crawled_unique = crawl_unique_links(
        links_with_index[link_column],
        checkpoint_dir=checkpoint_dir,
        timeout=timeout,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    result = links_with_index.merge(
        crawled_unique,
        on=link_column,
        how="left",
    )

    result = result.sort_values(
        original_index_column,
        kind="mergesort",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl article content from news links.")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--link-column", default=LINK_COLUMN)
    parser.add_argument("--original-index-column", default=ORIGINAL_INDEX_COLUMN)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = build_article_content_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        checkpoint_dir=args.checkpoint_dir,
        link_column=args.link_column,
        original_index_column=args.original_index_column,
        timeout=args.timeout,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    print(f"Saved article contents to: {args.output_path}")
    print("Rows:", len(result))
    print("Successful rows:", int(result["success"].fillna(False).sum()))
    print("Failed rows:", int((~result["success"].fillna(False)).sum()))


if __name__ == "__main__":
    main()
