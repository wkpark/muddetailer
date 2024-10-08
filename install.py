import os
import platform
import shutil
import sys

from packaging import version
from pathlib import Path
from typing import Tuple, Optional

import launch
from launch import is_installed, run, run_pip
import importlib.metadata

try:
    skip_install = getattr(launch.args, "skip_install")
except Exception:
    skip_install = getattr(launch, "skip_install", False)

python = sys.executable


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def install():
    if not is_installed("mim"):
        run_pip("install -U openmim", desc="openmim")

    # minimal requirement
    if not is_installed("mediapipe"):
        run_pip(f'install protobuf>=3.20')
        run_pip(f'install mediapipe>=0.10.3')

    torch_version = importlib.metadata.version("torch")
    legacy = None
    if torch_version:
        legacy = torch_version.split(".")[0] < "2"

    mmdet_v3 = None
    mmdet_version = None
    try:
        mmdet_version = importlib.metadata.version("mmdet")
    except Exception:
        pass
    if mmdet_version:
        mmdet_v3 = version.parse(mmdet_version) >= version.parse("3.0.0")

    if not is_installed("mmdet") or (legacy and mmdet_v3) or (not legacy and not mmdet_v3):
        if legacy:
            if mmdet_v3:
                print("Uninstalling mmdet, mmengine... (if installed)")
                run(f'"{python}" -m pip uninstall -y mmdet mmengine', live=True)
            run(f'"{python}" -m mim install mmcv-full', desc="Installing mmcv-full", errdesc="Couldn't install mmcv-full")
            run_pip(f"install mmdet==2.28.2", desc="mmdet")
        else:
            if not mmdet_v3:
                print("Uninstalling mmdet, mmcv, mmcv-full... (if installed)")
                run(f'"{python}" -m pip uninstall -y mmdet mmcv mmcv-full', live=True)
            print("Installing mmcv, mmdet, mmengine...")
            if not is_installed("mmengine"):
                run_pip(f"install mmengine==0.8.5", desc="mmengine")
            # mmyolo depends on mmcv==2.0.0 but pytorch 2.1.0 only work with mmcv 2.1.0
            if version.parse(torch_version) >= version.parse("2.2.0"):
                run(f'"{python}" -m mim install mmcv~=2.2.0', desc="Installing mmcv", errdesc="Couldn't install mmcv 2.2.0")
            elif version.parse(torch_version) >= version.parse("2.1.0"):
                run(f'"{python}" -m mim install mmcv~=2.1.0', desc="Installing mmcv", errdesc="Couldn't install mmcv 2.1.0")
            else:
                run(f'"{python}" -m mim install mmcv~=2.0.0', desc="Installing mmcv", errdesc="Couldn't install mmcv")
            run(f'"{python}" -m mim install -U mmdet>=3.0.0', desc="Installing mmdet", errdesc="Couldn't install mmdet")

            run_pip(f"install mmdet>=3", desc="mmdet")


    def fix_mmcv_version(filepath):
        """patch mmcv_maximum_version"""
        out = []
        ver = "0.0.0"
        with open(filepath, "r+") as f:
            while (line:= f.readline()):
                line = line.strip()
                if line.startswith("mmcv_maximum_version = "):
                    tmp = line.split("=")[1].strip("' ").split(".")
                    if len(tmp) == 3:
                        tag = tmp[2][1:]
                        tmp[2] = tmp[2][0]
                        if int(tmp[1]) < 2:
                            tmp[1] = str(int(tmp[1]) + 1)
                        tmp[2] = str(int(tmp[2]) + 1) + tag
                        ver = ".".join(tmp)
                        line = f"mmcv_maximum_version = '{ver}'\n"
                    else:
                        return ".".join(tmp)
                out.append(line.strip())

            f.seek(0)
            f.truncate(0)
            f.write("\n".join(out))
        return ver


    mmcv_version = None
    try:
        mmcv_version = importlib.metadata.version("mmcv")
    except Exception:
        pass
    if mmcv_version:
        print("Check mmcv version...")
        if version.parse(mmcv_version) >= version.parse("2.0.1"):
            print("need to check mmcv version in the __init__.py...")

            fixed_version = mmcv_version
            for p in sys.path:
                if os.path.exists(os.path.join(p, "mmyolo")) and os.path.exists(os.path.join(p, "mmyolo", "__init__.py")):
                    if not os.path.exists(os.path.join(p, "mmyolo", "__init__.py.bak")):
                        shutil.copy(os.path.join(p, "mmyolo", "__init__.py"), os.path.join(p, "mmyolo", "__init__.py.bak"))
                        fixed_version = fix_mmcv_version(os.path.join(p, "mmyolo", "__init__.py"))
                if os.path.exists(os.path.join(p, "mmdet")) and os.path.exists(os.path.join(p, "mmdet", "__init__.py")):
                    if not os.path.exists(os.path.join(p, "mmdet", "__init__.py.bak")):
                        shutil.copy(os.path.join(p, "mmdet", "__init__.py"), os.path.join(p, "mmdet", "__init__.py.bak"))
                        fixed_version = fix_mmcv_version(os.path.join(p, "mmdet", "__init__.py"))

            if fixed_version == mmcv_version:
                pass
            elif version.parse(fixed_version) > version.parse(mmcv_version):
                print("mmcv maximum version automatically fixed! (it may not work with the latest mmcv)")
            else:
                print(f"Your mmcv version {mmcv_version} may not work with the mmyolo/mmdet")
                print("or you need to fix version restriction of __init__.py of the mmyolo/mndet manually to use mmcv {mmcv_version}.")

    mmengine_version = None
    try:
        mmengine_version = importlib.metadata.version("mmengine")
    except Exception:
        pass
    if mmengine_version:
        print("Check mmengine version...")
        if version.parse(mmengine_version) >= version.parse("0.9.0") and version.parse(mmengine_version) < version.parse("0.10.0"):
            print(f"Your mmengine version {mmengine_version} may not work on windows...")
            print("Please install mmengine 0.8.5 manually or install latest bitsandbytes >= 0.43.0 or un-official patched version of bitsandbytes-windows.")
            #print("Uninstalling mmengine...")
            #run(f'"{python}" -m pip uninstall -y mmengine', live=True)
            #print("Installing mmengine 0.8.5...")
            #run_pip(f"install mmengine==0.8.5", desc="mmengine")
        else:
            print(f"your mmengine version is {mmengine_version}")

    if not legacy and not is_installed("mmyolo"):
        run(f'"{python}" -m pip install mmyolo', desc="Installing mmyolo", errdesc="Couldn't install mmyolo")


    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    if os.path.exists(req_file):
        mainpackage = 'uddetailer'
        with open(req_file) as file:
            for package in file:
                try:
                    package = package.strip()
                    if '==' in package:
                        package_name, package_version = package.split('==')
                        installed_version = get_installed_version(package_name)
                        if installed_version != package_version:
                            run_pip(f"install -U {package}", f"{mainpackage} requirement: changing {package_name} version from {installed_version} to {package_version}")
                    elif '>=' in package:
                        package_name, package_version = package.split('>=')
                        installed_version = get_installed_version(package_name)
                        if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                            run_pip(f"install -U {package}", f"{mainpackage} requirement: changing {package_name} version from {installed_version} to {package_version}")
                    elif not is_installed(package):
                        run_pip(f"install {package}", f"{mainpackage} requirement: {package}")
                except Exception as e:
                    print(f"Error: {e}")
                    print(f'Warning: Failed to install {package}, some preprocessors may not work.')


if not skip_install:
    install()
