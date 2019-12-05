# Release Checklist

When planning to release a new version of this package, it has proven effective to do the following:

* Update `setup.py` with version number and depencies
* Commit and push all changes
* `cd ~/temp`
* `pip uninstall` all dependencies
* Follow the installation instructions in README.md
* If successful, tag current diff with the new version number
* If not successful, fix the bug and start from the top
