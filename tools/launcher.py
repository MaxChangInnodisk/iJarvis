#!/usr/bin/python3
import concurrent.futures
import os
import time

import docker
import requests
from docker import errors
from rich.console import Console

DINO_CNTR = "gtc24_demo1.4"
LLM_CNTR = "llm"
JRVS_CNTR = "ijarvis"
WEBS_CNTR = "ijarvis-website"


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def verify_device():
    import subprocess as sp
    import sys

    bin_path = resource_path("./iSMART")
    cmd = f"{bin_path} -d /dev/nvme0n1 | grep EUI64"
    response = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    output, errors = response.communicate()
    if not output:
        # print("Verify Device ... FAILED ( Launch iSMART Failed )")
        sys.exit(1)
    ssd_code = output.split(":")[1].strip()
    if ssd_code != "24693E0000089634":
        # print(f"Verify Device ... FAILED ( Detected {ssd_code})")
        sys.exit(1)
    # print("Verify Device ... PASS")


def test_api(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出异常
        return True
    except requests.exceptions.RequestException:
        return False


def get_v4l2() -> list:
    return [f"/dev/video{i}" for i in range(10) if os.path.exists(f"/dev/video{i}")]


def get_displayer() -> list:
    if not os.environ.get("DISPLAY"):
        return []
    # Give Docker root user X11 permissions
    os.system("xhost +")
    # Enable SSH X11 forwarding inside container
    XAUTH = "/tmp/.docker.xauth"
    os.system(
        f"xauth nlist {os.environ['DISPLAY']} | sed -e 's/^..../ffff/' | xauth -f {XAUTH} nmerge -"
    )
    os.chmod(XAUTH, 0o777)
    return [
        f"-e DISPLAY={os.environ['DISPLAY']}",
        "-v /tmp/.X11-unix/:/tmp/.X11-unix",
        f"-v {XAUTH}:{XAUTH}",
        f"-e XAUTHORITY={XAUTH}",
    ]


# -------------


def run_vlm(client, test_host):
    vlm = client.containers.get(DINO_CNTR)
    if vlm.status != "running":
        vlm.start()
        # print("Start vlm")

    vlm.exec_run(["python3", "/data/haisten/GroundingDINO/main.py"], detach=True)

    ts = time.time()
    # print("Waitting for vlm ...")
    while not test_api(f"{test_host}:8000/docs"):
        time.sleep(1)
    # print(f"Launch vlm Service ... {int(time.time()-ts)}s")
    return vlm


def run_llm(client, test_host):
    try:
        llm = client.containers.get(LLM_CNTR)
        if llm.status == "running":
            llm.stop()
            # print("Stop previous llm")
    except errors.NotFound:
        pass
    with open("/proc/device-tree/model", "r") as model_file:
        nv_jetson_model = model_file.read().strip()
    container = client.containers.run(
        image="dustynv/text-generation-webui:r36.2.0",
        name=LLM_CNTR,
        runtime="nvidia",
        detach=True,
        network="haistenet",
        volumes={
            "/tmp/argus_socket": {"bind": "/tmp/argus_socket", "mode": "rw"},
            "/etc/enctune.conf": {"bind": "/etc/enctune.conf", "mode": "ro"},
            "/etc/nv_tegra_release": {"bind": "/etc/nv_tegra_release", "mode": "ro"},
            "/tmp/nv_jetson_model": {"bind": "/tmp/nv_jetson_model", "mode": "ro"},
            "/var/run/dbus": {"bind": "/var/run/dbus", "mode": "rw"},
            "/var/run/avahi-daemon/socket": {
                "bind": "/var/run/avahi-daemon/socket",
                "mode": "rw",
            },
            "/data/haisten/jetson-containers/data": {
                "bind": "/workspace",
                "mode": "rw",
            },
        },
        devices=["/dev/snd", "/dev/bus/usb"] + get_v4l2(),
        ports={"7860/tcp": 7860, "5000/tcp": 5000},
        environment=["DATA_VOLUME"] + get_displayer(),
        command='/bin/bash -c "cd /opt/text-generation-webui && python3 server.py \
            --model-dir=/workspace/models/text-generation-webui \
            --model=mistral-7b-v0.1.Q2_K.gguf \
            --loader=llamacpp \
            --n-gpu-layers=128 \
            --listen --chat --verbose \
            --listen-host 0.0.0.0 \
            --listen-port 7860 \
            --api-port 5000 \
            --api"',
        auto_remove=True,
    )

    ts = time.time()
    # print("Waitting for llm ...")
    while not test_api(f"{test_host}:5000/docs"):
        time.sleep(1)

    # print(f"Launch llm Service ... {int(time.time()-ts)}s")

    return container


def run_jarvis(client, test_host):
    jarvis = client.containers.get(JRVS_CNTR)
    if jarvis.status != "running":
        jarvis.start()
        # print("Start jarvis")

    jarvis.exec_run(["python3", "app.py"], detach=True)

    ts = time.time()
    # print("Waitting for jarvis ...")
    while not test_api(f"{test_host}:9527/docs"):
        time.sleep(1)
    # print(f"Launch jarvis Service ... {int(time.time()-ts)}s")
    return jarvis


def run_website(client, test_host):
    """
    sudo docker run -dt --name ijarvis-website \
    -e NGINX_PORT=8003 \
    -e BACKEND_PORT=9527 \
    -e BACKEND_SERVER=127.0.0.1 \
    --net host \
    -t innodiskorg/ijarvis-website:v101
    """
    jarvis_website = client.containers.get(WEBS_CNTR)
    if jarvis_website.status != "running":
        jarvis_website.start()
        # print("Start jarvis_website")

    ts = time.time()
    # print("Waitting for jarvis_website ...")
    while not test_api(f"{test_host}:8003"):
        time.sleep(1)
    # print(f"Launch jarvis_website Service ... {int(time.time()-ts)}s")
    return jarvis_website


def start_event():
    test_host = "http://127.0.0.1"
    client = docker.from_env()
    console = Console()

    futures = {}
    with console.status("[bold green]Launching Service...") as status:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures[executor.submit(run_llm, client, test_host)] = time.time()
            futures[executor.submit(run_vlm, client, test_host)] = time.time()
            futures[executor.submit(run_jarvis, client, test_host)] = time.time()
            futures[executor.submit(run_website, client, test_host)] = time.time()

            for future in concurrent.futures.as_completed(futures):
                created_time = futures[future]
                try:
                    container = future.result()
                    console.log(
                        f"{container.name} completed ({container.short_id}) ... cost {time.time()-created_time:.3f}s"
                    )
                except Exception as exc:
                    console.log(f"{container.name} generated an exception: {exc}")
    status.update("[bold green]All services processed", spinner="dots")


def stop_event():
    client = docker.from_env()
    console = Console()

    containers = []
    for name in (LLM_CNTR, DINO_CNTR, JRVS_CNTR, WEBS_CNTR):
        try:
            containers.append(client.containers.get(name))
        except errors.NotFound:
            pass

    futures = {}
    with console.status("[bold green]Stopping Service...") as status:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for container in containers:
                futures[executor.submit(container.stop)] = {
                    "name": container.name,
                    "created_time": time.time(),
                }

            for future in concurrent.futures.as_completed(futures):
                items = futures[future]
                container_name = items["name"]
                created_time = items["created_time"]
                try:
                    console.log(
                        f"{container_name} completed ... cost {time.time()-created_time:.3f}s "
                    )
                except Exception as exc:
                    console.log(f"{container_name} generated an exception: {exc}")
    status.update("[bold green]All services processed", spinner="dots")


if __name__ == "__main__":
    USAGE = "sudo ./executor [start|stop]"
    import sys

    assert len(sys.argv) > 1, USAGE
    option = sys.argv[1].lower()

    verify_device()

    if option == "start":
        start_event()

    elif option == "stop":
        stop_event()

    else:
        raise KeyError("Only support `start` and `stop`")
