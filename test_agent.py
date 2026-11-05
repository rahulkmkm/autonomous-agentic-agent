import tempfile
import unittest
from pathlib import Path

from autonomous_agent import AutonomousAgent, CalculatorTool, FileWriterTool, MockSearchTool


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_basic(self):
        self.assertEqual(self.calc.run("25 * 4 + 10").output, "110")
        self.assertEqual(self.calc.run("(12 + 8) * 5").output, "100")
        self.assertEqual(self.calc.run("7 / 2").output, "3.5")

    def test_rejects_code(self):
        for bad in ["__import__('os').system('echo hi')", "open('x')", "().__class__", "2 ** 9999"]:
            self.assertFalse(self.calc.run(bad).ok, bad)

    def test_divide_by_zero(self):
        self.assertFalse(self.calc.run("1/0").ok)


class FileWriterTests(unittest.TestCase):
    def test_sandboxed(self):
        with tempfile.TemporaryDirectory() as d:
            res = FileWriterTool(d).run('{"filename": "../../evil.txt", "content": "x"}')
            self.assertTrue(res.ok)
            self.assertTrue((Path(d) / "evil.txt").exists())

    def test_bad_json(self):
        self.assertFalse(FileWriterTool("x").run("not json").ok)


class SearchTests(unittest.TestCase):
    def test_hit_and_miss(self):
        s = MockSearchTool()
        self.assertTrue(s.run("AI agent trends 2026").ok)
        self.assertFalse(s.run("best pizza in rome").ok)


class AgentTests(unittest.TestCase):
    def make(self, d):
        return AutonomousAgent(verbose=False, output_dir=d)

    def test_calculate(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("1000", self.make(d).run("calculate 500 + 500"))

    def test_search_does_not_trigger_calculator_on_hyphen(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.make(d).run("search self-correction loops in AI agent trends 2026")
            self.assertNotIn("calculator", out)

    def test_multi_step_chain_saves_observations(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d).run("calculate (12 + 8) * 5, search AI agent trends 2026 and save the results to report.txt")
            text = (Path(d) / "report.txt").read_text()
            self.assertIn("100", text)
            self.assertIn("self-correction", text)

    def test_no_tool_needed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("did not find", self.make(d).run("hello there"))

    def test_step_limit(self):
        with tempfile.TemporaryDirectory() as d:
            agent = AutonomousAgent(verbose=False, output_dir=d, max_steps=1)
            self.assertIn("step limit", agent.run("calculate 1 + 1 and save to a.txt"))


if __name__ == "__main__":
    unittest.main()
