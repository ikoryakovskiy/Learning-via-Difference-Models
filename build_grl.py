import subprocess
p = subprocess.Popen(["python", "/grl/qt-build/cfg/build.py"], cwd="/grl/qt-build")
p.wait()
