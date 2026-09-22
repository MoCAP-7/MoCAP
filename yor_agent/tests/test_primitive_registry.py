from __future__ import annotations

import unittest

from yor_agent.primitives.registry import PrimitiveRegistry


def drive(distance_m: float, *, timeout_s: float | None = None) -> dict:
    """Drive a signed distance.

    Positive is forward.
    """

    return {"success": True, "distance_m": distance_m, "timeout_s": timeout_s}


class PrimitiveRegistryTest(unittest.TestCase):
    def test_functions_returns_a_detached_copy(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("drive", drive)

        functions = registry.functions()
        functions.pop("drive")

        self.assertIn("drive", registry)
        self.assertEqual(registry.names(), ["drive"])
        self.assertEqual(len(registry), 1)

    def test_documentation_exposes_signature_and_docstring(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("drive", drive)

        docs = registry.documentation()

        self.assertIn("def drive(distance_m: float, *, timeout_s: float | None = None)", docs)
        self.assertIn("Drive a signed distance.", docs)
        self.assertIn("Positive is forward.", docs)

    def test_rejects_duplicates_bad_names_and_non_callables(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("drive", drive)

        with self.assertRaises(ValueError):
            registry.register("drive", drive)
        with self.assertRaises(ValueError):
            registry.register("not an identifier", drive)
        with self.assertRaises(TypeError):
            registry.register("speed", 3.0)  # type: ignore[arg-type]

    def test_undocumented_callable_still_renders(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("noop", lambda: None)

        self.assertIn("(no documentation)", registry.documentation())


if __name__ == "__main__":
    unittest.main()
