"""Bridges to other simulation packages.

Each module here imports its target lazily, so none of them is a dependency of
htdse. Importing `htdse` never imports qutip; you pay only if you call the
bridge.
"""
