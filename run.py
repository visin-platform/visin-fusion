"""Compatibility launcher for existing SLURM jobs."""
import sys

from visin_fusion.cli import main, stage_command as stage_command

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
