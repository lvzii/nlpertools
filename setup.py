import os
import re

from setuptools import setup


def get_version():
    with open(os.path.join("src", "nlpertools", "__init__.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\'([^\"]+)\'".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


extra_require = {
    "torch": ["torch"],
}


def main():
    setup(
        install_requires=get_requires(),
        extras_require=extra_require,
        version=get_version()
    )


if __name__ == '__main__':
    main()
