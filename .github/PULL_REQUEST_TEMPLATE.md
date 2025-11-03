### Requirements
* Be aware of relevant documentation:
  * [Pull request documentation](https://hyperspy.org/rosettasciio/contributing.html#pull-requests)
  * [Adding new file format](https://hyperspy.org/rosettasciio/contributing.html#defining-new-rosettasciio-plugins)
  * [Adding and making test files](https://hyperspy.org/rosettasciio/contributing.html#making-test-data-files)
  * RosettaSciIO follows standard practises in open-source code development. If you need more information, you can refer to the [HyperSpy developer guide](https://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html) to familiarise yourself with how to contribute.
* Base your pull request on the ``main`` branch.
* Fill out the template; it helps the review process and it is useful to summarise the PR and its progress.
* Ask for help in the thread of the pull request when you need.
* Ask for review when you are ready.
* This template can be updated during the progression of the PR to summarise its status. 

*You can delete this section after you read it.*

### Description of the change
A few sentences and/or a bulleted list to describe and motivate the change:
- Change A.
- Change B.
- etc.

### Progress of the PR
- [ ] Change implemented (can be split into several points),
- [ ] update docstring (if appropriate),
- [ ] update user guide (if appropriate),
- [ ] add a changelog entry in the `upcoming_changes` folder (see [`upcoming_changes/README.rst`](https://github.com/hyperspy/rosettasciio/blob/main/upcoming_changes/README.rst)),
- [ ] Check formatting of the changelog entry (and eventual user guide changes) in the `docs/readthedocs.org:rosettasciio` build of this PR (link in github checks)
- [ ] add tests,
- [ ] ready for review.

### Minimal example of the bug fix or the new feature
```python
from rsciio.msa import file_reader
file_reader("your_msa_file.msa")
# Your new feature...
```
Note that this example can be useful to update the user guide.

