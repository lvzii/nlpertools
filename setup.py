import os
import re

from setuptools import setup


def get_version():
    with open(os.path.join("src", "nlpertools", "__init__.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\'([^\"]+)\'".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def main():
    setup(
        # https://juejin.cn/post/7369349560421040128
        install_requires=[
            "numpy",
            "pandas",
            "psutil"
        ],
        extras_require={
            "torch": ["torch"],
        },
        version=get_version(),
        entry_points={
            "console_scripts": [
                "ncli=nlpertools.cli:main",
            ]
        }
    )


if __name__ == '__main__':
    main()
