# 🎯 Pre-PyPI Publishing TODO List

## ⚠️ REQUIRED CHANGES BEFORE PUBLISHING

### 1. Update Email Addresses
Search for `your.email@example.com` and replace with your actual email in:
- [ ] `setup.py` (2 occurrences)
- [ ] `pyproject.toml` (2 occurrences)

**Command to find them:**
```bash
grep -r "your.email@example.com" setup.py pyproject.toml
```

### 2. Install Build Tools
```bash
pip install --upgrade build twine
```

### 3. Create PyPI Accounts
- [ ] Create Test PyPI account: https://test.pypi.org/account/register/
- [ ] Create PyPI account: https://pypi.org/account/register/
- [ ] Generate API token for Test PyPI
- [ ] Generate API token for PyPI

### 4. Verify Package Structure
```bash
# Run tests
pytest

# Check if main entry point works
python main.py --help
```

## ✅ READY TO PUBLISH

Once the above is complete, follow [PYPI_QUICKREF.md](./PYPI_QUICKREF.md) for quick commands or [PYPI_GUIDE.md](./PYPI_GUIDE.md) for detailed instructions.

## 📋 Publishing Checklist

- [ ] Update email addresses
- [ ] Set version to 0.1.0 (already done)
- [ ] All tests pass
- [ ] Build package: `python -m build`
- [ ] Test on Test PyPI
- [ ] Verify test installation works
- [ ] Publish to PyPI
- [ ] Create GitHub release
- [ ] Add PyPI badge to README

## 🔗 Next Steps After Publishing

1. **Add PyPI Badge to README.md:**
   ```markdown
   [![PyPI version](https://badge.fury.io/py/aibioagent.svg)](https://pypi.org/project/aibioagent/)
   ```

2. **Update Installation Instructions:**
   Users can now install with:
   ```bash
   pip install aibioagent
   ```

3. **Create GitHub Release:**
   - Go to GitHub → Releases → New Release
   - Tag: v0.1.0
   - Title: "AI Scientist v0.1.0 - Initial Release"
   - Description: Copy from CHANGELOG or summarize features

4. **Announce:**
   - Post on relevant forums (bioimaging, Python communities)
   - Share on social media
   - Consider submitting to JOSS (you already have PUBLISHING_GUIDE.md)

---

**Need help?** See the detailed guides:
- [PYPI_GUIDE.md](./PYPI_GUIDE.md) - Full publishing walkthrough
- [PYPI_QUICKREF.md](./PYPI_QUICKREF.md) - Quick command reference
