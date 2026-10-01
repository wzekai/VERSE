"""Dockerless SWE-rebench execution backend (optional; the paper's runs use docker).

Some hosts have no docker daemon but run as root with chroot available. A SWE-rebench task image
is a root filesystem with the repository pre-installed in a conda `testbed` env, so
`docker run <image> eval.sh` can be replaced by `chroot <exported rootfs> eval.sh`: the exported
rootfs gives a working testbed python, full pytest collection and a clean git tree.

Preparation, on a host with docker: export each image's rootfs
(`docker create <image>`, then `docker export <container> | zstd > <sanitized-image>.tar.zst`)
and upload it under T2E_TESTBED_S3. At run time, ensure_rootfs() downloads and extracts each
rootfs once into a local cache, and run_in_chroot() runs the eval script built by swe_judge with
the same log contract.

chroot isolates the filesystem only: there is no PID or network namespace, and commands run as
root. Each judge run uses a throwaway clone of the cached rootfs.

Environment variables:
    T2E_TESTBED_S3     s3://bucket/prefix holding <sanitized-image>.tar.zst (required)
    T2E_CHROOT_CACHE   local rootfs cache directory (default /opt/ml/t2e_testbeds if /opt/ml
                       exists, else ~/t2e_testbeds)
    T2E_CHROOT_FRESH   1 = re-extract the rootfs before every run (slow)
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile

logger = logging.getLogger("verse.runtime")


def sanitize(image: str) -> str:
    """Map an image ref to a key-safe name, e.g. 'swerebench/x:latest' -> 'swerebench__x__latest'.

    A ref without a tag gets ':latest' first, because the exported tarballs are always named with
    a tag."""
    if ":" not in image:
        image = image + ":latest"
    return re.sub(r"[/:]", "__", image)


def _cache_dir() -> str:
    d = os.environ.get("T2E_CHROOT_CACHE")
    if d:
        return d
    return "/opt/ml/t2e_testbeds" if os.path.isdir("/opt/ml") else os.path.expanduser("~/t2e_testbeds")


def ensure_rootfs(image: str) -> str | None:
    """Return a ready rootfs directory for `image`, downloading <s3>/<sanitized>.tar.zst once.

    Returns None if the tarball is missing or extraction fails; callers treat this like a docker
    failure. Safe when several workers or threads need the same image at once: a per-image
    flock serializes downloads across processes and threads (and avoids duplicate multi-GB
    downloads), and each download is extracted into a unique staging directory that one atomic
    os.rename publishes, so no worker sees a half-extracted tree."""
    root = os.path.join(_cache_dir(), sanitize(image))
    marker = os.path.join(root, ".t2e_ready")
    fresh = os.environ.get("T2E_CHROOT_FRESH", "0") == "1"
    if os.path.exists(marker) and not fresh:
        return root
    s3 = os.environ.get("T2E_TESTBED_S3", "").rstrip("/")
    if not s3:
        logger.warning("[chroot] T2E_TESTBED_S3 unset — cannot fetch testbed rootfs")
        return None
    import fcntl
    os.makedirs(_cache_dir(), exist_ok=True)
    lock_path = os.path.join(_cache_dir(), f".lock_{sanitize(image)}")
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)     # blocks until the concurrent fetcher finishes
    try:
        if os.path.exists(marker) and not fresh:
            return root                      # a concurrent worker fetched it while we waited
        return _fetch_rootfs(image, root, marker, fresh, s3)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _fetch_rootfs(image: str, root: str, marker: str, fresh: bool, s3: str) -> str | None:
    key = f"{s3}/{sanitize(image)}.tar.zst"
    uniq = f"{os.getpid()}_{os.urandom(4).hex()}"
    stage = os.path.join(_cache_dir(), f".stage_{sanitize(image)}_{uniq}")
    tmp_tar = stage + ".tar.zst"
    tmp_plain = stage + ".tar"
    try:
        os.makedirs(stage, exist_ok=True)
        # download with boto3 (the aws CLI may not be installed)
        import boto3
        m = re.match(r"s3://([^/]+)/(.+)", key)
        boto3.client("s3").download_file(m.group(1), m.group(2), tmp_tar)
        import zstandard
        import tarfile
        # Decompress to a seekable temp .tar first: hardlink members of docker-export tars make
        # tarfile seek backwards, which a streaming read ("r|" over stream_reader) does not allow.
        dctx = zstandard.ZstdDecompressor()
        with open(tmp_tar, "rb") as f, open(tmp_plain, "wb") as out:
            dctx.copy_stream(f, out)
        os.unlink(tmp_tar)
        with tarfile.open(tmp_plain, mode="r:") as tf:
            tf.extractall(stage)
        os.unlink(tmp_plain)
        open(os.path.join(stage, ".t2e_ready"), "w").write("ok")
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)   # replace a previous or partial extraction
        os.rename(stage, root)              # atomic publish (same filesystem)
        return root
    except Exception as e:
        logger.warning(f"[chroot] ensure_rootfs({image}) failed: {type(e).__name__}: {e}")
        return None
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        for p in (tmp_tar, tmp_plain):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass                          # already removed by a concurrent worker


def _mount_pseudo(root: str) -> list[str]:
    """Mount /proc, /dev, /dev/shm and /sys in the chroot, best effort.

    Many test suites need them (multiprocessing uses /dev/shm; psutil and similar read /proc).
    Returns the mountpoints that succeeded, for cleanup. Failures are not fatal: many suites
    pass without these filesystems, and some restricted containers deny mount."""
    mounted = []
    plans = [
        (["mount", "-t", "proc", "proc", os.path.join(root, "proc")], os.path.join(root, "proc")),
        (["mount", "--bind", "/dev", os.path.join(root, "dev")], os.path.join(root, "dev")),
        (["mount", "-t", "tmpfs", "shm", os.path.join(root, "dev", "shm")],
         os.path.join(root, "dev", "shm")),
        (["mount", "--bind", "/sys", os.path.join(root, "sys")], os.path.join(root, "sys")),
    ]
    for cmd, point in plans:
        try:
            os.makedirs(point, exist_ok=True)
            if subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0:
                mounted.append(point)
        except Exception:
            pass
    _ensure_dev_nodes(root)
    return mounted


def _ensure_dev_nodes(root: str) -> None:
    """Create the device nodes the eval path needs (null, zero, random, urandom, tty) with mknod.

    Some containers deny `mount` (no CAP_SYS_ADMIN) but allow mknod (CAP_MKNOD is a docker
    default). Without /dev/urandom, `git apply` fails with "unable to get random bytes".
    Raises RuntimeError if random or urandom cannot be created."""
    dev = os.path.join(root, "dev")
    os.makedirs(dev, exist_ok=True)
    nodes = [("null", 1, 3), ("zero", 1, 5), ("random", 1, 8), ("urandom", 1, 9), ("tty", 5, 0)]
    for name, major, minor in nodes:
        p = os.path.join(dev, name)
        if os.path.exists(p):
            continue
        r = subprocess.run(["mknod", "-m", "666", p, "c", str(major), str(minor)],
                           capture_output=True, timeout=15)
        if r.returncode != 0 and name in ("random", "urandom"):
            # a regular file as a stand-in would give every process the same bytes, so fail
            # instead
            raise RuntimeError(f"mknod {p} failed: {r.stderr.decode(errors='replace').strip()}")
    shm = os.path.join(dev, "shm")
    os.makedirs(shm, exist_ok=True)


def _umount(points: list[str]) -> None:
    for p in reversed(points):
        subprocess.run(["umount", "-l", p], capture_output=True, timeout=30)


def _clone_rootfs(base: str) -> str | None:
    """Make a throwaway clone of the cached rootfs for one judge run.

    Same recipe as SWEChrootSession.start: hardlink-clone the OS (`cp -al`, seconds, almost no
    extra disk) but make a real copy of /testbed, because the judge modifies it (git reset and
    apply, .pyc files, new test files). Returns the clone directory, or None on failure."""
    import uuid
    workdir = os.environ.get("T2E_SWE_WORKDIR", "/testbed").lstrip("/")
    clone = os.path.join(_cache_dir(), "_judge", uuid.uuid4().hex[:12])
    try:
        os.makedirs(os.path.dirname(clone), exist_ok=True)
        subprocess.run(["cp", "-al", base, clone], check=True, capture_output=True, timeout=300)
        tb = os.path.join(clone, workdir)
        subprocess.run(["rm", "-rf", tb], check=True, capture_output=True, timeout=120)
        subprocess.run(["cp", "-a", os.path.join(base, workdir), tb], check=True,
                       capture_output=True, timeout=600)
        return clone
    except Exception as e:
        logger.warning(f"[chroot] judge clone failed: {type(e).__name__}: {e}")
        subprocess.run(["rm", "-rf", clone], capture_output=True, timeout=120)
        return None


def run_in_chroot(image: str, eval_script: str, model_patch: str, test_patch: str,
                  timeout: int) -> str | None:
    """Run the swe_judge eval script in a chroot of `image`'s rootfs.

    Same contract as swe_judge._run_in_docker: returns the combined stdout and stderr, or None
    on an infrastructure failure. Each run uses a throwaway clone of the cached rootfs, never
    the shared cache: a test file added by `git apply test.patch` survives `git reset --hard`
    and would make the next run of the same task fail with "already exists in working
    directory". Like a fresh docker container, the clone also isolates concurrent runs."""
    base = ensure_rootfs(image)
    if base is None:
        return None
    root = _clone_rootfs(base)
    if root is None:
        return None
    t2e_dir = os.path.join(root, "t2e")
    os.makedirs(t2e_dir, exist_ok=True)
    mounted = _mount_pseudo(root)
    _ensure_dev_nodes(root)
    # docker writes resolv.conf and hosts at run time, and `docker export` leaves them empty.
    # Copy the host's files for working DNS (chroot shares the host network namespace).
    for etc in ("resolv.conf", "hosts"):
        try:
            shutil.copyfile(f"/etc/{etc}", os.path.join(root, "etc", etc))
        except Exception:
            pass
    # Remove cloud credentials, API keys and T2E_* settings from the environment. Unlike
    # `docker run`, chroot inherits the parent environment, so a variable such as AWS_PROFILE
    # would reach the graded tests, and tests that use boto3 (moto's mock AWS, twine, eodag)
    # would then fail with `ProfileNotFound` even on the reference patch. These tests are meant
    # to run without real credentials.
    clean_env = {k: v for k, v in os.environ.items()
                 if not (k.startswith("AWS_") or k.startswith("T2E_")
                         or k == "BEDROCK_REGION" or k.endswith("_API_KEY"))}
    clean_env["HOME"] = "/root"        # home inside the rootfs, not the host user's (~/.aws)
    try:
        open(os.path.join(t2e_dir, "model.patch"), "w").write(model_patch or "")
        open(os.path.join(t2e_dir, "test.patch"), "w").write(test_patch or "")
        open(os.path.join(t2e_dir, "eval.sh"), "w").write(eval_script)
        proc = subprocess.run(["chroot", root, "/bin/bash", "/t2e/eval.sh"],
                              capture_output=True, text=True, timeout=timeout, env=clean_env)
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        logger.warning(f"[chroot] eval timed out after {timeout}s for {image}")
        return None
    except Exception as e:
        logger.warning(f"[chroot] eval failed: {type(e).__name__}: {e}")
        return None
    finally:
        _umount(mounted)
        subprocess.run(["rm", "-rf", root], capture_output=True, timeout=120)
