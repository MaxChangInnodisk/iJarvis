

```bash
sudo -H pip install docker
sudo -H pip install pyinstaller
sudo -H pip install rich
```

```bash
pyinstaller -F --name ijarvis-launcher.bin --add-binary="./iSMART:." launcher.py
```