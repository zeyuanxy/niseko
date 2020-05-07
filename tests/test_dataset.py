"""Tests for NisekoDataset."""


def test_get_pipelines(context):
    dataset = context.get_dataset_by_id('1567_poker_hand')
    assert dataset
    dataset.get_pipelines()
