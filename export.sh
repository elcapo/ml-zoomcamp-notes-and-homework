#!/usr/bin/env bash

marimo export html -f notes.py -o results/notes.html
marimo export html -f module-1/notes.py -o results/module-1/notes.html
marimo export html -f module-1/homework.py -o results/module-1/homework.html
marimo export html -f module-2/notes.py -o results/module-2/notes.html
marimo export html -f module-2/homework.py -o results/module-2/homework.html
marimo export html -f module-3/notes.py -o results/module-3/notes.html
marimo export html -f module-3/homework.py -o results/module-3/homework.html