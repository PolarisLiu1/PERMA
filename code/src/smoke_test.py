import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def run_cmd(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": " ".join(cmd),
        "code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=str, default=".env")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--mem-frame", type=str, default="supermemory")
    parser.add_argument("--output-dir", type=str, default="../../data/evaluation_smoke")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()

    if not env_path.exists():
        print(f"ENV_NOT_FOUND: {env_path}")
        sys.exit(2)

    load_dotenv(env_path)

    required_env = [
        "CHAT_MODEL",
        "CHAT_MODEL_API_KEY",
        "CHAT_MODEL_BASE_URL",
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        print("ENV_MISSING_KEYS:", ", ".join(missing))
        sys.exit(3)

    modules = [
        "complete_dataset_generator",
        "evaluation",
        "prompt",
        "function.client",
        "function.ingestion",
        "function.search",
    ]
    import_failures = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            import_failures.append((m, str(e)))

    if import_failures:
        print("IMPORT_FAILED")
        for m, e in import_failures:
            print(f"{m}: {e}")
        sys.exit(4)

    checks = []
    checks.append(run_cmd([sys.executable, "complete_dataset_generator.py", "--help"], str(base_dir)))
    checks.append(run_cmd([sys.executable, "evaluation.py", "--help"], str(base_dir)))

    if args.run_eval:
        checks.append(
            run_cmd(
                [
                    sys.executable,
                    "evaluation.py",
                    "--mode",
                    "baseline",
                    "--mem_frame",
                    args.mem_frame,
                    "--stage",
                    "add",
                    "search",
                    "--output_dir",
                    args.output_dir,
                    "--user_id_truncate",
                    "1",
                    "--num_workers",
                    "1",
                    "--debug",
                    "True",
                ],
                str(base_dir),
            )
        )

    failed = [c for c in checks if c["code"] != 0]
    print("SMOKE_TEST_SUMMARY")
    for c in checks:
        print(f"[{c['code']}] {c['cmd']}")
        if c["code"] != 0:
            if c["stderr"]:
                print(c["stderr"][:1200])
            elif c["stdout"]:
                print(c["stdout"][:1200])

    if failed:
        sys.exit(5)
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
