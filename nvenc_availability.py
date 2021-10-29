# MIT License
#
# Copyright (c) 2021 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re as _re
import logging as _logging
import warnings as _warnings
import subprocess as _sp
from collections import namedtuple as _namedtuple

_logging.basicConfig(level=_logging.INFO,
                     format="[%(asctime)s %(name)s] %(levelname)s: %(message)s")
LOGGER = _logging.getLogger(__name__)
LOGGER.setLevel(_logging.INFO)

def find_command(cmd):
    proc = _sp.run(["where", cmd], shell=True,
                   capture_output=True)
    if proc.returncode != 0:
        _warnings.warn(f"failed to find the '{cmd}' command: 'where' returned code {proc.returncode}")
        return None

    commands = [item.strip() for item in proc.stdout.decode().split("\n")]
    if len(commands) == 0:
        _warnings.warn(f"the '{cmd}' command not found")
        return None
    return commands[0]

class NvidiaGPU(_namedtuple("_NvidiaGPU", ("index", "name", "uuid"))):
    NAME_PATTERN = _re.compile(r"GPU (\d+): ([a-zA-Z0-9 -]+) \(UUID: ([a-zA-Z0-9-]+)\)")
    THREADS_PER_CORE = 3
    import nvenc_dataset as NVENC

    @classmethod
    def parse(cls, line):
        matches = cls.NAME_PATTERN.match(line.strip())
        if not matches:
            LOGGER.warning(f"failed to parse: {line.strip()}")
            return cls(-1, "<unknown>", "0")
        index = int(matches.group(1))
        name  = str(matches.group(2))
        uuid  = str(matches.group(3))
        return cls(index, name, uuid)

    def has_nvenc_h264(self):
        if ("Quadro" in self.name) or ("NVS" in self.name) or ("Tesla" in self.name):
            ## FIXME
            LOGGER.warning("note that having this type of GPUs does not ascertain the availability of NVEnc functionality: " \
                         "check the list of supported GPUs at: https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new")
            return True
        elif self.name in self.NVENC.NONE.keys():
            LOGGER.warning(f"{self.name} appears to have no NVENC core.")
            return False
        elif self.name in self.NVENC.AMBIG.keys():
            LOGGER.warning(f"unable to determine whether {self.name} has an NVENC core. NVENC functionality may not work properly.")
            return True
        elif self.name not in self.NVENC.COMPAT.keys():
            LOGGER.warning(f"{self.name} is not registered in the list of NVENC-compatible NVIDIA cores. This may mean the core is either too old or too new. Consult the developer if you think it is a bug.")
            return False
        else:
            n_cores = [self.NVENC.NVENC_CORES_H264.get(coretype, 0) for coretype in self.NVENC.COMPAT[self.name]]
            min_cores = min(n_cores)
            max_cores = max(n_cores)
            if min_cores == max_cores:
                LOGGER.info(f"{self.name}: number of NVENC cores: {max_cores} " \
                             f"(up to {max_cores*self.THREADS_PER_CORE} simultaneous encoding)")
            else:
                LOGGER.info(f"{self.name}: number of NVENC cores: {min_cores}-{max_cores}" \
                             f"(depends; possibly up to {max_cores*self.THREADS_PER_CORE} simultaneous encoding)")
            if min_cores == 0:
                LOGGER.warning(f"note that {self.name} may _not_ have an NVENC-compatible core, and the functionality may not work properly.")
            return max_cores > 0

def number_of_nvenc_gpus():
    nvidia_smi = find_command('nvidia-smi')
    if nvidia_smi is None:
        return 0  # cannot detect NVIDA driver

    # check the version of the driver
    proc = _sp.run([nvidia_smi], shell=True, capture_output=True)
    if proc.returncode != 0:
        _warnings.warn(f"failed to run 'nvidia-smi': code {proc.returncode}")
        return 0
    out = [line.strip() for line in proc.stdout.decode().split("\n") if len(line.strip()) > 0]
    pattern = _re.compile(r"Driver Version: (\d+(\.\d+)?)") # trying to capture only 'xx.yy', instead of 'xx.yy.zz'
    version = None
    for line in out:
        matched = pattern.search(line)
        if matched:
            version = matched.group(1)
            break
    if version is None:
        LOGGER.warning("failed to parse the driver version from 'nvidia-smi'")
        return 0
    major, minor = [int(digits) for digits in version.split(".")]
    LOGGER.debug(f"NVIDIA driver version: {version} (major={major}, minor={minor})")
    if major < 450:
        LOGGER.warning("NVIDIA driver must be newer than 450.xx: please update via https://www.nvidia.com/Download/driverResults.aspx/176854/en-us")
        return 0

    # check the number of GPUs abailable
    proc = _sp.run([nvidia_smi, "-L"], shell=True,
                   capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode().strip()
        _warnings.warn(f"failed to list up available GPUs: 'nvidia-smi' returned code {proc.returncode} ({err})")
        return 0

    GPUs = [NvidiaGPU.parse(item.strip()) for item in proc.stdout.decode().split("\n") if len(item.strip()) > 0]
    if len(GPUs) > 0:
        names = f" ({', '.join(core.name for core in GPUs)})"
    else:
        names = ""
    LOGGER.info(f"number of NVIDIA GPUs available: {len(GPUs)}{names}")
    return len([core for core in GPUs if core.has_nvenc_h264()])
