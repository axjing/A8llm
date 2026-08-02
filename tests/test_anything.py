import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.logger import print0, print_banner

print0("Hello World!")
print_banner()
