# Publishing Native Installers to GitHub Releases

When a new version tag (e.g., v1.2.3) is pushed to the repository, the GitHub Actions workflow in `.github/workflows/build-native.yml` will:

1. Build native installers for Windows, macOS, and Linux using PyInstaller.
2. Upload the resulting artifacts to the GitHub Release for that tag.

## How it works
- On tag push, the workflow builds the app for each OS and uploads the contents of the `dist/` directory as release assets.
- Users can download the appropriate installer from the "Releases" page on GitHub.

## Manual Release Steps (if needed)
If you need to manually upload an installer:
1. Build the app locally using PyInstaller.
   ```bash
   python -m PyInstaller --noconfirm --onedir --windowed --name pyopenms-viewer \
	   --collect-all pyopenms --collect-all plotly --additional-hooks-dir=. \
	   pyopenms_viewer/__main__.py
   ```
2. Go to the GitHub Releases page for your repo.
3. Click "Edit" on the relevant release, and upload the installer file(s).

## Notes
- Artifacts are unsigned by default. See TROUBLESHOOTING.md for platform-specific notes on unsigned apps.
- For code signing and notarization, see the next section.
- Each release should reference the latest CI smoke tests for Windows and Linux. The workflows in `.github/workflows/windows.yml` and `.github/workflows/linux.yml` build on clean GitHub-hosted runners and execute `pyopenms-viewer --help` to catch missing DLL/SO issues. Link to those successful runs in the release description so downstream users can see they were validated on native runners.

## Optional Code Signing & Notarization (CI)

The `Build and Release Native Installers` workflow automatically signs binaries when the relevant secrets are configured. Without secrets, the workflow behaves exactly as before and ships unsigned installers.

### Windows (Authenticode)
1. Export your Authenticode certificate to `codesign.pfx` and base64-encode it:
   ```bash
   base64 -i codesign.pfx | pbcopy
   ```
2. Add these repository secrets:
   - `WINDOWS_CERT_BASE64`
   - `WINDOWS_CERT_PASSWORD`
3. The workflow imports the certificate on the runner and executes:
   ```powershell
   signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 dist/pyopenms-viewer.exe
   ```

### macOS (codesign + notarytool)
1. Export your Developer ID Application certificate as `.p12`, base64-encode it, and configure secrets:
   - `APPLE_CERT_BASE64`
   - `APPLE_CERT_PASSWORD`
   - `APPLE_TEAM_ID`
   - `APPLE_NOTARIZATION_APPLE_ID`
   - `APPLE_NOTARIZATION_APP_PASSWORD`
2. The workflow creates a temporary keychain, imports the certificate, signs `dist/pyopenms-viewer.app`, runs `xcrun notarytool submit ... --wait`, staples the ticket, and re-zips the bundle.

### Release checklist
- After secrets are set, verify signatures with `signtool verify /pa dist/pyopenms-viewer.exe` and `codesign -dv --verbose=4 dist/pyopenms-viewer.app`.
- Mention in the GitHub release notes whether the binaries were signed/notarized and link to the smoke-test runs for traceability.
