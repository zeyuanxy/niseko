"""Niseko tests configurations and fixtures."""

import os

import pytest

from niseko import NisekoContext


@pytest.fixture(scope='session')
def context():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "niseko-dumps")
    context = NisekoContext(data_dir)
    yield context
