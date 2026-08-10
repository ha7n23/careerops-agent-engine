"""Exit successfully only when the CareerOps API reports ready."""

from urllib.error import URLError
from urllib.request import urlopen

READINESS_URL = "http://127.0.0.1:8000/ready"

READINESS_TIMEOUT_SECONDS = 3.0


def main() -> None:
    """Call the local readiness endpoint for container health checking."""

    try:
        with urlopen(
            READINESS_URL,
            timeout=(READINESS_TIMEOUT_SECONDS),
        ) as response:
            if response.status != 200:
                raise SystemExit(1)

    except (
        URLError,
        TimeoutError,
    ) as exc:
        print(f"CareerOps readiness check failed: {exc}")

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
