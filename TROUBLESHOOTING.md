# Native Installer Troubleshooting

## Windows
- If the app fails to launch, ensure Microsoft Edge WebView2 runtime is installed (required for pywebview).
- If you see missing DLL errors, install the Visual C++ Redistributable.
- SmartScreen warnings go away once the `.exe` is signed. Configure the `WINDOWS_CERT_BASE64` and `WINDOWS_CERT_PASSWORD` secrets so CI can sign releases automatically.

## macOS
- If you see a warning about unsigned apps, right-click the .app and choose "Open" to bypass Gatekeeper.
- If Finder reports that the app is damaged or it just bounces in the dock, remove the quarantine flag after unzipping:
	```bash
	xattr -dr com.apple.quarantine pyopenms-viewer.app
	```
- If the app fails to launch, check for missing modules or .dylib dependencies in the Console log. Recent builds bundle Plotly, but if you see `ModuleNotFoundError: plotly` reinstall the latest release or rebuild with the documented PyInstaller command.
- For full notarization configure the `APPLE_CERT_*` and `APPLE_NOTARIZATION_*` secrets so the automated build signs, notarizes, and staples the `.app` before release. Otherwise follow the Gatekeeper bypass steps above.

## Linux
- If the AppImage does not launch, ensure you have GTK and WebKit2 runtime libraries installed.
- Run `ldd` on the binary to check for missing shared libraries.

## General
- If native file dialogs do not appear, ensure pywebview is bundled and working.
- For browser fallback, run with `--browser` instead of `--native`.
- If you encounter issues, please open an issue on GitHub with your OS and error details.
