"""Shared fixtures for the test suite."""

from routewright_ai.engine import RoutewrightEngine


def engine():
    return RoutewrightEngine.shared(log=lambda *a: None)
