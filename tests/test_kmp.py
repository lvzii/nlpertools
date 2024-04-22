#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import pytest as pytest


@pytest.mark.parametrize("main_string, pattern_string, ans", [("a", "a", 0)])
def test_kmp_find_after_build(main_string, pattern_string, ans):
    from src.nlpertools.algo import kmp

    res = kmp.find_after_build(main_string, pattern_string)
    assert res == ans


def test_ml_calc_llm_train_activation_memory():
    from src.nlpertools.ml import calc_llm_train_activation_memory

    res = calc_llm_train_activation_memory(
        model_name="",
        sequence_length=2048,
        batch_size=1,
        hidden_dim=4096,
        lay_number=28,
        attention_heads_num=32,
        gpu_num=1,
    )
    print(res, "G")


test_ml_calc_llm_train_activation_memory()
