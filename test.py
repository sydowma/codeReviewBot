import os
import unittest

from main import Review


INPUT1 = """
"""

INPUT2 = """
"""

class MyTestCase(unittest.TestCase):
    def test_something(self):
        """
            This is a test function
            :return:
            """
        os.environ.setdefault('all_proxy', '')
        os.environ.setdefault('http_proxy', '')
        os.environ.setdefault('https_proxy', '')
        os.putenv('OPENAI_HTTP_PROXY', '')
        review = Review()
        review_result = review.parse_review_result(INPUT2)
        print(review_result)


if __name__ == '__main__':
    unittest.main()
