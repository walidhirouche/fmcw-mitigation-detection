"""Validation-only selection, with the earliest exact F1 tie."""
from protocol import select_epoch


def rank_epochs(history, targets, scenes):
    result=select_epoch(history,targets,scenes)
    return dict(epoch=result['epoch'],tp=result['TP'],fp=result['FP'],fn=result['FN'],F1=result['F1'])
