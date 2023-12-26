from src.nlpertools.io.dir import case_sensitive_path_exists

test_cases = [
    r"D:\S",
    r"D:\s",
    r"D:\a",
    r"D:\A",
]
for case in test_cases:
    print(case_sensitive_path_exists(case, False))
test_cases = ["S", "s", "a", "A"]
for case in test_cases:
    print(case_sensitive_path_exists(case, True))
