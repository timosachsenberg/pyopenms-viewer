# PyInstaller hook for pyopenms
# Collect the entire package (dylibs, data files, hidden imports) so that the
# bundled macOS app ships every dependency required by libOpenMS and Qt.

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('pyopenms')

