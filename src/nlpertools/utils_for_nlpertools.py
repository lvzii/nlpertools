import os
import shutil
from importlib import import_module

from .io.dir import j_mkdir
from .io.file import readtxt_list_all_strip, writetxt_w_list


def try_import(name, package):
    try:
        return import_module(name, package=package)
    except:
        print("import {} failed".format(name))
    finally:
        pass


def convert_import_to_try_import(from_path, to_path):
    j_mkdir(to_path)
    for root, dirs, files in os.walk(from_path):
        for sub_dir in dirs:
            j_mkdir(os.path.join(root.replace(from_path, to_path), sub_dir))
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(root.replace(from_path, to_path), file)
            excluded_file = ["wrapper.py", "kmp.py", "__init__.py"]
            if file.endswith(".py") and file != "utils_for_nlpertools.py" and file not in excluded_file:
                raw_code = readtxt_list_all_strip(src)
                start_idx, end_idx = 0, 0

                for idx, each_line in enumerate(raw_code[:30]):
                    each_line = each_line.lstrip("# ")
                    if start_idx == 0 and (each_line.startswith("from") or each_line.startswith("import")):
                        try:
                            exec(each_line)
                        except:
                            start_idx = idx
                    if start_idx != 0 and not each_line:
                        end_idx = idx
                        break
                # print(file, start_idx, end_idx)
                if start_idx != 0 and end_idx != 0:
                    new_code = raw_code[:start_idx] + convert_import_string_to_import_list(
                        "\n".join(raw_code[start_idx:end_idx])) + raw_code[end_idx:]
                else:
                    new_code = raw_code
                writetxt_w_list(new_code, dst)
            else:
                shutil.copy(src=src, dst=dst)
    print("convert over")


def get_import_info(text):
    pass


def convert_import_string_to_import_list(text):
    """
    该方法将 import 转变为 try import
    """
    models_to_import = []
    import_list = text.split("\n")
    for each in import_list:
        print(each)
        name, package, as_name = None, None, None
        elements = each.split(" ")
        for pre, cur in zip(elements, elements[1:]):
            if cur.endswith(","):
                cur = cur.rstrip(",")
            # 为了实现from import 和 import统一，首先把package和name的含义反过来，后面再掉换
            if pre == "import":
                package = cur
            if pre == "from":
                name = cur
            if pre == "as":
                as_name = cur
            if pre[-1] == ",":
                # 针对 from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
                # 将将前面部分和当前的组成新字段
                prefix = each.split("import")[0]
                import_list.append("{}import {}".format(prefix, cur))
        if not as_name:
            as_name = package.split(".")[-1]
        if not name:
            name, package = package, name
        models_to_import.append((name, package, as_name))
    # 打印
    all_import_info = ["", "from utils_for_nlpertools import try_import", ""]
    for name, package, as_name in models_to_import:
        import_info = '{} = try_import("{}", {})'.format(as_name, name, '"{}"'.format(package) if package else package)
        all_import_info.append(import_info)
        print(import_info)
    return all_import_info
