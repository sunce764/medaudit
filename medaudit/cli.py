"""Command-line entry point for medaudit.

    medaudit version
    medaudit audit --config audit.json [--out report.txt]
"""
import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="medaudit",
        description="Reliability audit toolkit for medical-image classifiers.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="Print the installed version.")

    a = sub.add_parser("audit", help="Run a reliability audit from a config file.")
    a.add_argument("--config", required=True, help="Path to the audit config (JSON).")
    a.add_argument("--out", default="report.txt", help="Output report path.")

    args = parser.parse_args(argv)

    if args.command == "version":
        from medaudit import __version__
        print(__version__)
        return 0
    if args.command == "audit":
        from medaudit.audit import run_audit, render_report
        report = run_audit(args.config)
        text = render_report(report)
        print(text)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n[medaudit] report written to {args.out}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
