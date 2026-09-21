"""System snapshot: CPU, memory, temperature, battery, power plan. Works with or without psutil."""
import os
import platform
import subprocess


def snapshot():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil:
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        f = psutil.cpu_freq()
        if f:
            info["cpu_mhz"] = round(f.current)
            info["cpu_max_mhz"] = round(f.max) if f.max else None
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 2 ** 30, 1)
        info["ram_used_percent"] = vm.percent
        b = getattr(psutil, "sensors_battery", lambda: None)()
        if b:
            info["battery_percent"] = round(b.percent)
            info["plugged"] = b.power_plugged
        temps = getattr(psutil, "sensors_temperatures", lambda: {})()
        if temps:
            first = next(iter(temps.values()))
            if first:
                info["temp_c"] = first[0].current
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            info["power_plan"] = out.split("(")[-1].rstrip(")") if "(" in out else out
        except Exception:
            pass
    return info
