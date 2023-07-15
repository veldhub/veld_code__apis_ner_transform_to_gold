import sys
import os
import datetime


print(f"datetime.datetime.now(): {datetime.datetime.now()}")
print(f"sys.version: {sys.version}")
print(f"sys.executable: {sys.executable}")
print(f"os.geteuid(): {os.geteuid()}")
print(f"os.getcwd(): {os.getcwd()}")
print(f"os.listdir(): {os.listdir()}")

