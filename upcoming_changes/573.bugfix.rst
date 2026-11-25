Loading a NeXus file containing a hard-link cycle between groups no longer recurses until the recursion limit is hit; each group reachable through more than one hard link is now walked once.
