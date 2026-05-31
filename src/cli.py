from __future__ import annotations

import argparse
import sys

from ru_liquidity_sentinel.logging import configure_logging, get_logger
from ru_liquidity_sentinel.pipeline.runner import Pipeline

logger = get_logger(__name__)

def main(argv: list[str] | None=None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog='sentinel', description='RU Liquidity Sentinel: liquidity stress early warning (PSB)')
    sub = parser.add_subparsers(dest='command', required=True)
    run_p = sub.add_parser('run', help='Fetch data, compute modules and LSI')
    run_p.add_argument('--no-cache', action='store_true', help='Force refresh HTTP cache')
    sub.add_parser('dashboard', help='Launch Streamlit dashboard')
    args = parser.parse_args(argv)
    if args.command == 'run':
        try:
            Pipeline().run()
            logger.info('success')
            return 0
        except Exception as exc:
            logger.exception('pipeline_failed', error=str(exc))
            return 1
    if args.command == 'dashboard':
        import subprocess
        from pathlib import Path
        app = Path(__file__).resolve().parents[2] / 'dashboard' / 'app.py'
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(app)], check=False)
        return 0
    return 1
if __name__ == '__main__':
    raise SystemExit(main())
