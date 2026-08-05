#!/usr/bin/env python3
"""Smoke-test a running mlflow models serve endpoint with a POST /invocations.

Cross-platform: uses requests, not curl. Prints status, response body (or error).
Validates payload format against the documented six options.

Usage:
    # JSON with dataframe_split (default)
    python scripts/probe_endpoint.py --url http://127.0.0.1:5001/invocations \\
        --input sample.json --format dataframe_split

    # CSV
    python scripts/probe_endpoint.py --url http://127.0.0.1:5001/invocations \\
        --input sample.csv --format csv

    # Just health-check
    python scripts/probe_endpoint.py --url http://127.0.0.1:5001/health --mode health

Exit codes:
    0  Endpoint responded successfully (any 2xx).
    1  Endpoint returned 4xx or 5xx.
    2  Connection refused / timeout / import error.
    3  Usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EXIT_OK = 0
EXIT_HTTP_ERROR = 1
EXIT_NETWORK = 2
EXIT_USAGE = 3


def send_request(url: str, payload: Any, fmt: str, timeout: float) -> tuple[int, Any]:
    """Send POST with appropriate content-type based on fmt. Returns (status_code, body)."""
    import requests

    if fmt == "csv":
        with open(payload, "rb") as f:
            r = requests.post(url, data=f.read(),
                              headers={"Content-Type": "text/csv"},
                              timeout=timeout)
    else:
        with open(payload) as f:
            if fmt == "dataframe_split":
                data = json.load(f)
            elif fmt == "dataframe_records":
                data = json.load(f)
            elif fmt in ("instances", "inputs"):
                data = json.load(f)
            else:
                print(f"Error: unknown format '{fmt}'", file=sys.stderr)
                sys.exit(EXIT_USAGE)
        r = requests.post(url, json=data, timeout=timeout)

    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="Endpoint URL (e.g. http://127.0.0.1:5001/invocations)")
    parser.add_argument("--input", help="Path to payload file (JSON or CSV depending on --format)")
    parser.add_argument("--format", choices=["dataframe_split", "dataframe_records", "instances", "inputs", "csv"],
                        default="dataframe_split",
                        help="Payload format (default: dataframe_split)")
    parser.add_argument("--mode", choices=["invocations", "health"], default="invocations",
                        help="Use 'health' to just check /health endpoint")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.mode == "invocations" and not args.input:
        print("Error: --input is required for mode=invocations", file=sys.stderr)
        return EXIT_USAGE

    if args.mode == "health":
        import requests
        try:
            r = requests.get(args.url, timeout=args.timeout)
            print(f"GET {args.url} → {r.status_code}")
            print(f"Body: {r.text[:500]}")
            return EXIT_OK if r.status_code == 200 else EXIT_HTTP_ERROR
        except requests.exceptions.ConnectionError:
            print(f"Error: connection refused at {args.url}", file=sys.stderr)
            return EXIT_NETWORK
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
            return EXIT_NETWORK

    try:
        status, body = send_request(args.url, args.input, args.format, args.timeout)
    except FileNotFoundError:
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:
        # requests.exceptions.ConnectionError / Timeout / SSLError all live here
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_NETWORK

    print(f"POST {args.url} ({args.format}) → {status}")
    if isinstance(body, (dict, list)):
        print(json.dumps(body, indent=2)[:1500])
    else:
        print(f"Body: {str(body)[:1500]}")

    return EXIT_OK if 200 <= status < 300 else EXIT_HTTP_ERROR


if __name__ == "__main__":
    sys.exit(main())