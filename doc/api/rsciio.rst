:mod:`rsciio`
=============

.. currentmodule:: rsciio

.. autosummary::
   :signatures: none

   __version__
   IO_PLUGINS
   set_log_level

.. Use py:data instead of autodata for __version__ because it's a string,
   and autodata would show the str class docstring instead.

.. py:data:: __version__
   :type: str

   The version of the RosettaSciIO package.

.. IO_PLUGINS is loaded lazily, so we need to use autodata here.

.. autodata:: IO_PLUGINS

.. automodule:: rsciio
  :members:
