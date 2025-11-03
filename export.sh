#!/usr/bin/env bash

source .venv/bin/activate

marimo export html -f notes.py -o results/notes.html
marimo export html -f module-1/notes.py -o results/module-1/notes.html
marimo export html -f module-1/homework.py -o results/module-1/homework.html
marimo export html -f module-2/notes.py -o results/module-2/notes.html
marimo export html -f module-2/homework.py -o results/module-2/homework.html
marimo export html -f module-3/notes.py -o results/module-3/notes.html
marimo export html -f module-3/homework.py -o results/module-3/homework.html
marimo export html -f module-4/notes.py -o results/module-4/notes.html
marimo export html -f module-4/homework.py -o results/module-4/homework.html
marimo export html -f module-5/notes.py -o results/module-5/notes.html
marimo export html -f module-5/homework.py -o results/module-5/homework.html
marimo export html -f module-6/notes.py -o results/module-6/notes.html

