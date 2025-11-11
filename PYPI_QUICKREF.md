# 📦 PyPI Quick Reference

## 🏗️ Build Package
```bash
# Clean previous builds
rm -rf build dist *.egg-info

# Build
python -m build
```

## 🧪 Test PyPI
```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aibioagent
```

## 🚀 Production PyPI
```bash
# Upload to PyPI
python -m twine upload dist/*

# Install command for users
pip install aibioagent
```

## ✅ Pre-publish Checklist
- [ ] Update version in `setup.py` and `pyproject.toml`
- [ ] Update email addresses (search for TODO)
- [ ] Run tests: `pytest`
- [ ] Clean build: `rm -rf build dist *.egg-info`
- [ ] Build: `python -m build`
- [ ] Test on Test PyPI first
- [ ] Tag release: `git tag -a v0.1.0 -m "Release 0.1.0"`

## 📝 Version Updates
```bash
# In setup.py
version="0.2.0"

# In pyproject.toml
version = "0.2.0"

# Git tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

## 🔧 Install Build Tools
```bash
pip install --upgrade build twine
```

For full details, see [PYPI_GUIDE.md](./PYPI_GUIDE.md)
