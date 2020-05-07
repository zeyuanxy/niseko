"""Tests for NisekoContext."""


def test_list_datasets(context):
    datasets = context.list_datasets()
    assert len(datasets) == 300


def test_get_dataset_by_id(context):
    dataset = context.get_dataset_by_id('1567_poker_hand')
    assert dataset
    dataset.show_stats()
