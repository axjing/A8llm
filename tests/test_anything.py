import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlm.common.logger import DummySwanLab, init_swanlab, print0, print_banner

print0("Hello World!")
print_banner()


def test_init_swanlab_disabled_returns_dummy() -> None:
    run = init_swanlab(
        project="proj",
        run_name="run",
        config={"lr": 1e-4},
        enabled=False,
    )
    assert isinstance(run, DummySwanLab)
    run.log({"loss": 0.1}, step=1)
    run.finish()
