import io

from setuptools import find_packages
from setuptools import setup

with io.open("README.rst", "rt", encoding="utf8") as f:
    readme = f.read()

setup(
    name="vgac_tagging",
    version="0.1.0",
    url="https://pom-itb-gitlab01.campus.pomona.edu/faim-lab/vgac_tagging",
    license="BSD",
    maintainer="FAIM Lab team",
    maintainer_email="gbkh2015@mymail.pomona.edu",
    description="Current tagging tool for VGAC",
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["flask"],
    extras_require={"test": ["pytest", "coverage"]},
)
