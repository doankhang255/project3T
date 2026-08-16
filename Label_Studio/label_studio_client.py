"""Kết nối tới Label Studio API bằng token trong .env và liệt kê project."""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ["LABEL_STUDIO_URL"].rstrip("/")
REFRESH_TOKEN = os.environ["LABEL_STUDIO_TOKEN"]


def get_access_token() -> str:
    resp = requests.post(
        f"{BASE_URL}/api/token/refresh/",
        json={"refresh": REFRESH_TOKEN},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access"]


def list_projects(access_token: str) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/api/projects/",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["results"] if isinstance(data, dict) else data


def main() -> None:
    try:
        access_token = get_access_token()
    except requests.HTTPError as e:
        print(f"Không lấy được access token: {e} - {e.response.text}", file=sys.stderr)
        sys.exit(1)

    projects = list_projects(access_token)
    if not projects:
        print("Không tìm thấy project nào.")
        return

    for p in projects:
        print(f"[{p['id']}] {p['title']} - tasks: {p.get('task_number', '?')}")


if __name__ == "__main__":
    main()
