"""Remove a seated USB plug and return it to its tabletop home.

Public runtime types live in ``setup``, ``task`` and ``execution``.  Keeping the
package initializer import-free avoids a cycle while shared scene configuration
imports this task's ``config`` module.
"""
