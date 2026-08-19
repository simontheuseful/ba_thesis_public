import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DIAGRAMS_DIR = os.path.join(_HERE, "..", "diagrams")

DEFAULT_VOLUME = "cloud_462"

SCRIPTS = {
    "compare_render_time_transmittance.py": "{volume}.pdf",
    "compare_render_time_transmittance_dda.py": "{volume}_dda.pdf",
    "compare_render_time_random.py": "{volume}_random.pdf",
    "compare_render_time_linear.py": "{volume}_linear.pdf",
}


def main(volume_name=None):
    volume_name = volume_name or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOLUME)

    out_dir = os.path.join(DIAGRAMS_DIR, volume_name)
    os.makedirs(out_dir, exist_ok=True)

    for script, pdf_pattern in SCRIPTS.items():
        script_path = os.path.join(_HERE, script)
        print(f"\n{'=' * 80}\nRunning {script} {volume_name}\n{'=' * 80}")
        result = subprocess.run([sys.executable, script_path, volume_name])

        pdf_name = pdf_pattern.format(volume=volume_name)
        src = os.path.join(DIAGRAMS_DIR, pdf_name)

        dst = os.path.join(out_dir, pdf_name)
        shutil.move(src, dst)


if __name__ == "__main__":
    main()
