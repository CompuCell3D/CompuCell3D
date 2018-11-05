Adding New Setting
==================
To add new setting you need to edit
<CC3D_REPOSITORY>/CompuCell3D/player5/Configuration_settings/_settings.sqlite
and
<CC3D_REPOSITORY>/CompuCell3D/player5/Configuration_settings/osx/_settings.sqlite

to include the new settings.

Use SQLLiteBrowser program to do that.

Any new settings will be synchronized in the

.. code-block:: python

    synchronizeGlobalAndDefaultSettings(default_settings, global_settings, global_settings_path)

inside **Configuration/SettingUtils.py**