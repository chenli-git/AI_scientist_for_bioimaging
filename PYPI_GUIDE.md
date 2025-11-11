# 📦 Publishing to PyPI

This guide walks you through publishing `aibioagent` to the Python Package Index (PyPI).

## 📋 Prerequisites

1. **PyPI Account**: Create accounts on both [Test PyPI](https://test.pypi.org/account/register/) and [PyPI](https://pypi.org/account/register/)
2. **API Tokens**: Generate API tokens for authentication (more secure than passwords)
3. **Build Tools**: Install required packages

```bash
pip install --upgrade build twine
```

## 🔧 Pre-Publishing Checklist

Before publishing, ensure:

- [ ] **Version number** is updated in `setup.py` and `pyproject.toml`
- [ ] **Email address** is updated in `setup.py` and `pyproject.toml` (search for TODO comments)
- [ ] All tests pass: `pytest`
- [ ] README.md is complete and well-formatted
- [ ] LICENSE file exists
- [ ] `.gitignore` excludes build artifacts

## 🏗️ Step 1: Build the Package

Clean any previous builds and create distribution files:

```bash
# Remove old builds
rm -rf build dist *.egg-info

# Build the package
python -m build
```

This creates two files in the `dist/` directory:
- `aibioagent-0.1.0.tar.gz` (source distribution)
- `aibioagent-0.1.0-py3-none-any.whl` (wheel distribution)

## 🧪 Step 2: Test on Test PyPI (Recommended)

Always test on Test PyPI first to catch issues before the real release.

### Upload to Test PyPI

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your Test PyPI API token (starts with `pypi-`)

### Test Installation

Create a fresh virtual environment and test install:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aibioagent

# Test basic import
python -c "from agents.AI_scientist_agent import AIScientistAgent; print('Import successful!')"

# Deactivate and clean up
deactivate
rm -rf test_env
```

**Note**: The `--extra-index-url` is needed because Test PyPI doesn't have all dependencies.

## 🚀 Step 3: Publish to PyPI

Once you've verified everything works on Test PyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

## 📥 Installation by Users

After publishing, users can install with:

```bash
pip install aibioagent
```

Or with development dependencies:

```bash
pip install aibioagent[dev]
```

## 🔑 Setting Up API Tokens

### For Test PyPI
1. Go to https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Set scope to "Entire account" (or specific project later)
4. Copy the token (starts with `pypi-`)

### For PyPI
1. Go to https://pypi.org/manage/account/token/
2. Follow same steps as Test PyPI

### Save Tokens in `.pypirc` (Optional)

Create `~/.pypirc` to avoid entering tokens manually:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Important**: Keep this file secure (`chmod 600 ~/.pypirc`)

## 🔄 Publishing Updates

When releasing a new version:

1. **Update version number** in:
   - `setup.py` (line with `version="0.1.0"`)
   - `pyproject.toml` (line with `version = "0.1.0"`)

2. **Update CHANGELOG** (create one if needed):
   ```markdown
   ## [0.2.0] - 2025-01-15
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

3. **Tag the release** in git:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

4. **Rebuild and upload**:
   ```bash
   rm -rf build dist *.egg-info
   python -m build
   python -m twine upload dist/*
   ```

## 🎯 Version Numbering (Semantic Versioning)

Follow [semver.org](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fixes
- `0.1.0` → `0.2.0`: New feature (Paper Reviewer Agent)
- `0.9.0` → `1.0.0`: Stable release, API frozen

## 🧹 Cleanup After Publishing

```bash
# Remove build artifacts (don't commit these)
rm -rf build dist *.egg-info

# Or add to .gitignore:
echo "build/" >> .gitignore
echo "dist/" >> .gitignore
echo "*.egg-info/" >> .gitignore
```

## ❌ Common Issues

### Issue: "File already exists"
**Solution**: You cannot re-upload the same version. Increment version number.

### Issue: "Invalid distribution filename"
**Solution**: Ensure version numbers match in `setup.py` and `pyproject.toml`.

### Issue: Dependencies not installing
**Solution**: Check that version constraints aren't too restrictive. Use `>=X.Y.Z,<MAJOR+1.0.0` format.

### Issue: "Module not found" after install
**Solution**: 
- Verify `__init__.py` exists in all package directories
- Check `packages=find_packages()` in `setup.py` is correct
- Ensure package name uses hyphens (`aibioagent`) but imports use underscores if needed

## 📊 Package Statistics

After publishing, you can track:
- **Downloads**: https://pypistats.org/packages/aibioagent
- **Package page**: https://pypi.org/project/aibioagent/

## 🔗 Useful Links

- [PyPI](https://pypi.org/)
- [Test PyPI](https://test.pypi.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## 🎓 Next Steps

After publishing to PyPI:
1. Add PyPI badge to README: `[![PyPI version](https://badge.fury.io/py/aibioagent.svg)](https://badge.fury.io/py/aibioagent)`
2. Announce on social media/forums
3. Create GitHub release matching the version tag
4. Update documentation with installation instructions
5. Consider submitting to JOSS for academic citation

---

**Ready to publish?** Start with Test PyPI, verify everything works, then publish to PyPI! 🚀
