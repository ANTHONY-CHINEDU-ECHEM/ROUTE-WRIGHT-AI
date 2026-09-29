import unittest

from routewright_ai.style import scan
from routewright_ai.utils import FORBIDDEN_CHARS, HYPHEN, clean_dashes, contains_dash

EM = chr(8212)
EN = chr(8211)


class HouseStyleTests(unittest.TestCase):
    def test_project_contains_no_hyphens_or_dashes(self):
        problems = scan()
        self.assertEqual(problems, [], "Found dash characters: {}".format(problems[:5]))

    def test_clean_dashes_joined_words(self):
        self.assertEqual(clean_dashes("well" + HYPHEN + "defined plan"), "well defined plan")

    def test_clean_dashes_em_and_en(self):
        text = clean_dashes("Budget" + EM + "the real issue" + EN + "slipped")
        self.assertFalse(contains_dash(text))

    def test_clean_dashes_spaced_and_bullets(self):
        text = clean_dashes(HYPHEN + " first\n" + HYPHEN + " second\nA " + EM + " B")
        self.assertTrue(text.startswith("* first"))
        self.assertIn("A, B", text)

    def test_clean_dashes_negative_numbers(self):
        self.assertEqual(clean_dashes("variance of " + HYPHEN + "5"), "variance of minus 5")

    def test_all_forbidden_characters_removed(self):
        text = "a".join(FORBIDDEN_CHARS) + "b"
        self.assertFalse(contains_dash(clean_dashes(text)))


if __name__ == "__main__":
    unittest.main()
