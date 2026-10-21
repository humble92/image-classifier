"""Convenience wrapper: ``python calibrate.py LABELED_DIR --refs ...``."""

import sys

from image_classifier.cli import app

if __name__ == "__main__":
    app(["calibrate", *sys.argv[1:]])
