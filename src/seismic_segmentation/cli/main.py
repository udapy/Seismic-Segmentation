# src/seismic_segmentation/cli/main.py
"""Main entry point for the seismic segmentation CLI."""

import argparse
import importlib
import sys
from pathlib import Path

import yaml


def main():
    """Main entry point for seismic segmentation CLI."""
    parser = argparse.ArgumentParser(description="Seismic Segmentation Pipeline")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return 1

    # Add debug flag to config
    config["debug"] = args.debug

    # Get task
    task = config.get("task", "train")

    # Import appropriate module based on task
    try:
        module = importlib.import_module(f"seismic_segmentation.tasks.{task}")
        # Run the task
        result = module.run(config)
        return 0 if result else 1
    except ImportError as e:
        print(f"Error: Task module '{task}' not found: {e}")
        return 1
    except Exception as e:
        if config.get("debug", False):
            # In debug mode, show full traceback
            import traceback

            traceback.print_exc()
        else:
            print(f"Error executing task '{task}': {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
