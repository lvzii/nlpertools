#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import pytest as pytest

from src.nlpertools.algo import kmp
@pytest.mark.parametrize(
    "main_string, pattern_string, ans",
    [
        ("a", "a", 0)
    ]
)
def test_kmp_find_after_build(main_string, pattern_string, ans):

    res = kmp.find_after_build(main_string, pattern_string)
    assert res == ans

