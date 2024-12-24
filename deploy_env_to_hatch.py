from pathlib import Path
import subprocess
import sys


def load_dotenv(env_path=None):
    """Load environment variables from .env file"""
    if env_path is None:
        env_path = Path('.env')

    if not env_path.exists():
        print(f"Error: {env_path} file not found")
        return False

    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                except ValueError:
                    print(f"Warning: Invalid line in .env file: {line}")
                    continue

    return env_vars


def deploy_to_hatch(env_vars):
    """Deploy environment variables to Hatch environment"""
    for key, value in env_vars.items():
        try:
            subprocess.run(
                ['hatch', 'config', 'set', f'env.default.env.{key}', value],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Successfully set {key} in Hatch environment")
        except subprocess.CalledProcessError as e:
            print(f"Error setting {key}: {e.stderr}")
            return False
    return True


def main():
    env_vars = load_dotenv()
    if not env_vars:
        sys.exit(1)

    if deploy_to_hatch(env_vars):
        print("Successfully deployed all environment variables to Hatch")
    else:
        print("Error deploying environment variables")
        sys.exit(1)


if __name__ == "__main__":
    main()
