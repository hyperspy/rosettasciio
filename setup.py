"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
import os
import warnings

from setuptools import Command, Extension, setup

setup_path = os.path.abspath(os.path.dirname(__file__))


# Extensions. Add your extension here:
raw_extensions = [
    Extension(
        "rsciio.bruker.unbcf_fast", [os.path.join("rsciio", "bruker", "unbcf_fast.pyx")]
    ),
]

cleanup_list = []
for leftover in raw_extensions:
    path, ext = os.path.splitext(leftover.sources[0])
    if ext in (".pyx", ".py"):
        cleanup_list.append("".join([os.path.join(setup_path, path), ".c*"]))
        if os.name == "nt":
            bin_ext = ".cpython-*.pyd"
        else:
            bin_ext = ".cpython-*.so"
        cleanup_list.append("".join([os.path.join(setup_path, path), bin_ext]))


def count_c_extensions(extensions):
    c_num = 0
    for extension in extensions:
        # if first source file with extension *.c or *.cpp exists
        # it is cythonised or pure c/c++ extension:
        sfile = extension.sources[0]
        path, ext = os.path.splitext(sfile)
        if os.path.exists(path + ".c") or os.path.exists(path + ".cpp"):
            c_num += 1
    return c_num


def cythonize_extensions(extensions):
    try:
        from Cython.Build import cythonize

        cythonize(extensions, language_level="3")
    except Exception:
        warnings.warn(
            """WARNING: The compilation of the C-extension failed.
            Only slow pure python alternative functions will be available.
            To use fast implementation of some functions writen in cython/c either:
            a) check that you have compatible compiler installed,
            b) cython is installed.
            b) use binary distribution of hyperspy (i.e. wheels, conda packages).
            Installation will continue in 5 sec..."""
        )
        from time import sleep

        sleep(5)

    return []


def no_cythonize(extensions):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


# to cythonize, or not to cythonize... :
if os.environ.get("DISABLE_C_EXTENTIONS"):
    # Explicitly disable
    extensions = []
elif len(raw_extensions) > count_c_extensions(raw_extensions):
    extensions = cythonize_extensions(raw_extensions)
else:
    extensions = no_cythonize(raw_extensions)


class Recythonize(Command):
    """cythonize all extensions"""

    description = "(re-)cythonize all changed cython extensions"

    user_options = []

    def initialize_options(self):
        """init options"""
        pass

    def finalize_options(self):
        """finalize options"""
        pass

    def run(self):
        global extensions
        cythonize_extensions(extensions)


setup(
    ext_modules=extensions,
    cmdclass={
        "recythonize": Recythonize,
    },
)
