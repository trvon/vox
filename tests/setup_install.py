"""Exercise setup.sh without downloading models or touching the real user manager."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SETUP = Path(__file__).resolve().parents[1] / "setup.sh"


class SetupInstall(unittest.TestCase):
    def run_setup(self, extra_env=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "mock-bin"
            bin_dir.mkdir()
            scripts = {
                "uname": "#!/bin/sh\necho Linux\n",
                "pkg-config": "#!/bin/sh\nexit 0\n",
                "systemctl": '#!/bin/sh\n[ "${NO_USER_MANAGER:-0}" = 1 ] && exit 1\nexit 0\n',
                "vox": '#!/bin/sh\nprintf "%s\\n" "$*" >> "$INSTALL_LOG"\n',
                "cargo": '#!/bin/sh\nmkdir -p target/release\ncp "$MOCK_VOX" target/release/vox\n',
            }
            for name, content in scripts.items():
                script = bin_dir / name
                script.write_text(content)
                script.chmod(0o755)
            shutil.copyfile(SETUP, root / "setup.sh")
            env = {
                **os.environ,
                "HOME": str(root),
                "XDG_DATA_HOME": str(root / "data"),
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                "MOCK_VOX": str(bin_dir / "vox"),
                "INSTALL_LOG": str(root / "install.log"),
                "VOX_NO_SERVICE": "0",
                **(extra_env or {}),
            }
            result = subprocess.run(
                ["bash", "setup.sh", str(root / "install with spaces")],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.access(root / "install with spaces/vox", os.X_OK))
            return (root / "install.log").read_text().splitlines(), result.stdout

    def test_source_install_enables_service(self):
        calls, _ = self.run_setup()
        self.assertEqual(calls, ["download-models", "service install"])

    def test_opt_out_does_not_install_service(self):
        calls, output = self.run_setup({"VOX_NO_SERVICE": "1"})
        self.assertEqual(calls, ["download-models"])
        self.assertIn("Skipping autostart", output)

    def test_no_user_manager_retains_binary_and_models(self):
        calls, output = self.run_setup({"NO_USER_MANAGER": "1"})
        self.assertEqual(calls, ["download-models"])
        self.assertIn("No systemd user manager available", output)


if __name__ == "__main__":
    unittest.main()
