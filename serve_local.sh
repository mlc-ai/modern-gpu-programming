#!/bin/bash
d2lbook build html
python -m http.server -d _build/html
